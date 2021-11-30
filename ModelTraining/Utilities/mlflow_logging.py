from mlflow import log_artifacts, log_metric, log_param

def logParams(args, seed):

    log_param("seed", str(seed))
    log_param("PICO entity", args.entity)
    log_param("input file", args.rawcand_file)
    log_param("embedding", args.embed)
    log_param("model name", args.model)
    log_param("label type", args.label_type)
    log_param("training data", args.train_data)
    log_param("epochs", args.max_eps)
    log_param("learning rate", args.lr)
    log_param("lr warmup", args.lr_warmup)
    log_param("max seq length", args.max_len) 
    log_param("Transformer Frozen", args.freeze_bert)


def logMetrics(name, metric_value, step):
    log_metric(key=name, value=metric_value, step=step)


def logIntermediateMetrics(name, metric_value):
    log_metric(key=name, value=metric_value)