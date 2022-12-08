from math import cos, pi

from torch.optim import Adam
from torch.optim import SGD

from networks.PRGG import ProgReprGestureGen
from networks.EncodeAudio import AudioEncoder
from networks.GenerateMotion import Generator
from networks.Discriminator import GesDiscriminator

class Custom_Optimizer(object):
    def __init__(self, optim_type, model, init_lr, epoches): 
        self.model = model 
        model_type = self.identify_model(model)
        self.foundation_module_name_head = self.get_foundation_name(model_type)
        self.progressive_module_name_head = self.get_progressive_name(model_type)
        
        self.moduleName_parameters = dict(self.model.named_parameters())
        self.moduleName_idx = {k : i for i, k in enumerate(self.moduleName_parameters.keys())}
        self.idx_moduleName = {i : k for k, i in self.moduleName_idx.items()}

        self.optim = self.get_target_optim(optim_type)
        
        self.epoches = epoches 
        self.total_epoches = sum(self.epoches)
        self.init_lr = init_lr 

        self.init_optim()
        
    def get_foundation_name(self, model_type):
        foundation = [] 
        if model_type == 'gen': # generator 
            self.load_names(foundation, AudioEncoder.foundation_name)
            self.load_names(foundation, Generator.foundation_name)
            
        else : # discriminator
            self.load_names(foundation, GesDiscriminator.foundation_name)

        return foundation

    def get_progressive_name(self, model_type):
        progressive = [] 
        if model_type == 'gen': # generator 
            self.load_names(progressive, AudioEncoder.progressive_name)
            self.load_names(progressive, Generator.progressive_name)
            
        else : # discriminator
            self.load_names(progressive, GesDiscriminator.progressive_name)

        return progressive

    def load_names(self, name_list, target): 
        if isinstance(target, list): 
            name_list.extend(target)
        else :
            name_list.append(target)

    def init_optim(self):
            optim_empty = True 
            for moduleName, param in self.moduleName_parameters.items():
                if optim_empty : 
                    self.optim = self.optim([param], lr=self.init_lr)
                    optim_empty = False 
                else : 
                    self.optim.add_param_group(param_group={'params' : [param], 'lr' : self.init_lr})

    def adjust_learning_rate(self, cur_step, cur_total_epoch): 
        for step in range(cur_step, -1, -1): 
            target_lr = self.get_target_cos_lr(step, cur_total_epoch)
            adjust_idx = self.get_adjust_idx(step)
            for idx in adjust_idx : 
                self.adjust_lr_by_idx(target_lr, idx)

    def get_target_cos_lr(self, step, cur_total_epoch):
        max_iter = self.total_epoches - ( sum(self.epoches[:step]) - 1 )
        cur_iter = cur_total_epoch - ( sum(self.epoches[:step]) - 1 )
        target_lr = self.init_lr * (1 + cos(pi * cur_iter / max_iter)) / 2
        return target_lr 

    def get_adjust_idx(self, cur_step): 
        adjust_idx = []
        adjust_module_names = [name_head + f'.{cur_step}' for name_head in self.progressive_module_name_head]
        adjust_module_names += self.foundation_module_name_head
        for adjust_moule_name in adjust_module_names: 
            for module_name, idx in self.moduleName_idx.items():
                if adjust_moule_name in module_name:
                    adjust_idx.append(idx)
        return list(set(adjust_idx))

    def adjust_lr_by_idx(self, target_lr, idx):
        self.optim.param_groups[idx]['lr'] = target_lr

    def step_optim(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def get_target_optim(self, optim):
        if optim == 'Adam':
            return Adam
        elif optim == 'SGD' : 
            return SGD
        else : 
            raise ValueError(f'Only optim is either Adam or SGD')
    
    def identify_model(self, model): 
        if isinstance(model, ProgReprGestureGen): 
            return 'gen'
        elif isinstance(model, GesDiscriminator): 
            return 'dis'
        else : 
            raise ValueError('model shoulde be ethier ProgReprGestureGen or GesDiscriminator')