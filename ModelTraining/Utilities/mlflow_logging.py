from mlflow import log_artifacts, log_metric, log_param

def logParams(args):

    log_param("input file", args.input_file)
    log_param("embedding", args.embed)
    log_param("model name", args.model)
    log_param("label type", args.label_type)
    log_param("training data", args.train_data)
    log_param("epochs", args.max_eps)
    log_param("learning rate", args.lr)
    log_param("max seq length", args.max_len) 