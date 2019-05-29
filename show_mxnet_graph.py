import mxnet as mx

def load_model_sym(mprefix, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=mprefix,
                                                            epoch=epoch)    
    return sym

if __name__ == "__main__":
    sym = load_model_sym("./model_mxnet/model", 0)
    data_shape = {"data":(1,3,112,112)}
    net_show = mx.viz.plot_network(symbol=sym,shape=data_shape)  
    net_show.render(filename="mxnet_mobile",cleanup=True)
