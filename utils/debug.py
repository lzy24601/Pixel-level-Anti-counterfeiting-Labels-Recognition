def show_gradstate(model):
    for name, parms in model.named_parameters():
        if name == "module.conv1.weight":	
            print('-->name:', name)
            print('-->grad_requirs:',parms.requires_grad)
            print('-->grad_value:',parms.grad)
            print("===========================")
        
def freeze_para(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
