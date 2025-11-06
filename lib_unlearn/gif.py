import time
import torch
from torch.autograd import grad

class GIF_Unlearn():
    def __init__(self, target_model, args):
        self.target_model = target_model
        self.args = args

    def gif_approxi(self, res_tuple, evaluate_F1):
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''
        start_time = time.time()
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']

        if self.args["method"] == "GIF" or self.args["method"] == "CEU":
            v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        if self.args["method"] == "IF":
            v = res_tuple[1]
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        #input(h_estimate)
        for _ in range(iteration):

            model_params  = [p for p in self.target_model.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            #input(hv)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
                #input(h_estimate)
                #input([torch.sum(torch.isinf(h)) for h in h_estimate])
        params_change = [h_est / scale for h_est in h_estimate]
        #input(scale)
        #input(params_change)
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        #print(params_change)
        #input()
        if evaluate_F1:
            test_F1 = self.target_model.evaluate_unlearn_F1(params_esti)
        else:
            test_F1 = -1
        return time.time() - start_time, test_F1, params_esti

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
    

class GA_Unlearn():
    def __init__(self, target_model, args):
        self.target_model = target_model
        self.args = args

    def unlearn(self):
        start_time = time.time()
        self.target_model.train_grad_ascent()
        new_parameters = [p for p in self.target_model.model.parameters() if p.requires_grad]
        test_F1 = self.target_model.evaluate_unlearn_F1(new_parameters)
        
        return time.time() - start_time, test_F1, new_parameters