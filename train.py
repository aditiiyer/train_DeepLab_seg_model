import sys

import json
import numpy as np
import os
from dataloaders.custom_dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.calculate_weights import calculate_weights_labels
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary

from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback


class Trainer(object):

    def __init__(self, args, modelConfig, inputH5Path):

        # Get training parameters
        hyperpars = args["hyperparameters"]
        archpars = args["architecture"]

        # Get model config
        structList = modelConfig["structList"]
        nclass = len(structList) + 1  # + 1 for background class

        args["nclass"] = nclass
        args["inputH5Path"] = inputH5Path
        if torch.cuda.device_count() and torch.cuda.is_available():
            print('Using GPU...')
            args["cuda"] = True
            deviceCount = torch.cuda.device_count()
            print('GPU device count: ', deviceCount)
        else:
            print('using CPU...')
            args["cuda"] = False

        # Use default args where missing
        defPars = {'fineTune': False, 'resumeFromCheckpoint': None, 'validate': True,
                   'evalInterval': 1}
        defHyperpars = {'startEpoch': 0}
        for key in defPars.keys():
            if not key in args.keys():
                args[key] = defPars[key]
        for key in defHyperpars.keys():
            if not key in hyperpars.keys():
                hyperpars[key] = defHyperpars[key]

        args["hyperparameters"] = hyperpars
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloaders
        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_set = customData(self.args, split='Train')
        self.train_loader = DataLoader(train_set, batch_size=hyperpars["batchSize"], shuffle=True, drop_last=True,
                                       **kwargs)
        val_set = customData(self.args, split='Val')
        self.val_loader = DataLoader(val_set, batch_size=hyperpars["batchSize"], shuffle=False, drop_last=False,
                                     **kwargs)
        test_set = customData(self.args, split='Test')
        self.test_loader = DataLoader(test_set, batch_size=hyperpars["batchSize"], shuffle=False, drop_last=False,
                                      **kwargs)

        # Define network
        model = DeepLab(num_classes=args["nclass"],
                        backbone='resnet',
                        output_stride=archpars["outStride"],
                        sync_bn=archpars["sync_bn"],
                        freeze_bn=archpars["freeze_bn"],
                        model_path=args["modelSavePath"])

        train_params = [{'params': model.get_1x_lr_params(), 'lr': hyperpars["lr"]},
                        {'params': model.get_10x_lr_params(), 'lr': hyperpars["lr"] * 10}]

        # Define Optimizer
        optimizer_type = args["optimizer"]
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(train_params, momentum=hyperpars["momentum"],
                                        weight_decay=hyperpars["weightDecay"], nesterov=hyperpars["nesterov"])
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(train_params, lr=hyperpars["lr"],
                                         betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=hyperpars["weightDecay"])

        # Initialize weights
        print('Initializing weights...')
        initWeights = args["initWeights"]
        if initWeights["method"] == "classBalanced":
            # Use class balanced weights
            print('Using class-balanced weights.')
            class_weights_path = os.path.join(inputH5Path, 'classWeights.npy')
            if os.path.isfile(class_weights_path):
                print('reading weights from' + class_weights_path)
                weight = np.load(class_weights_path)
            else:
                weight = calculate_weights_labels(inputH5Path, self.train_loader, args["nclass"])
                np.save(class_weights_path, weight)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        # Define loss function
        self.criterion = SegmentationLosses(weight=weight, cuda=args["cuda"]).build_loss(mode=args["lossType"])
        self.model, self.optimizer = model, optimizer

        # Define evaluator
        self.evaluator = Evaluator(args["nclass"])

        # Define lr scheduler
        self.scheduler = LR_Scheduler(hyperpars["lrScheduler"], hyperpars["lr"],
                                      hyperpars["maxEpochs"], len(self.train_loader))

        # Use GPU(s) if available
        if args["cuda"]:
            self.model = torch.nn.DataParallel(self.model, list(range(deviceCount)))
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resume from previous checkpoint
        self.best_pred = 0.0
        if args["resumeFromCheckpoint"] is not None:
            if not os.path.isfile(args["resumeFromCheckpoint"]):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args["startEpoch"] = checkpoint['epoch']
            if args["cuda"]:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            # For fine-tuning:
            if not args["fineTune"]:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args["fineTune"]:
            args["startEpoch"] = 0

    def training(self, epoch):
        args = self.args
        hyperpars = args["hyperparameters"]
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if args["cuda"]:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show inference results
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, args, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * hyperpars["batchSize"] + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if not args["validate"]:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print('Best model yet!')  # AI temp
            try:
                state_dict = self.model.module.state_dict()
            except AttributeError:
                state_dict = self.model.state_dict()
            # end mod
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        if ((epoch + 1) == self.args.epochs):
            is_best = False
            try:
                state_dict = self.model.module.state_dict()
            except AttributeError:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'filename': 'last_checkpoint.pth.tar',
            }, is_best)


def loadJSONConfig(trainingParamFile):
    # inputJSONPath = os.path.join('/scratch',trainingParamFile)
    inputJSONPath = trainingParamFile
    f = open(inputJSONPath, )
    config = json.load(f)
    return config


def main():
    trainingParamFile = sys.argv[1]
    modelConfigFile = sys.argv[2]
    inputH5Dir = sys.argv[3]

    # Read hyperparameters from JSON input
    print('Reading training parameters...')
    trainParams = loadJSONConfig(trainingParamFile)
    print('Reading model configuration...')
    modelConfig = loadJSONConfig(modelConfigFile)

    # Train model
    torch.manual_seed(1)
    trainer = Trainer(trainParams, modelConfig, inputH5Dir)
    hyperpars = trainer.args["hyperparameters"]
    evalInterval = trainer.args["evalInterval"]
    print('Starting Epoch:', hyperpars["startEpoch"])
    print('Total Epoches:', hyperpars["maxEpochs"])
    for epoch in range(hyperpars["startEpoch"], hyperpars["maxEpochs"]):
        trainer.training(epoch)
    if not trainer.args["validate"] and epoch % evalInterval == (evalInterval - 1):
        trainer.validation(epoch)

    trainer.writer.close()


# if __name__ == "__main__":
main()
