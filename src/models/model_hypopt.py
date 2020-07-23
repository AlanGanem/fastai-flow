import sherpa
from tqdm.notebook import tqdm

SHERPA_PARAMETER_TYPE_MAPPER = {
    'choice':sherpa.Choice,
    'continuous':sherpa.Continuous,
    'discrete':sherpa.Discrete,
}
CLASS_SUFIX_MAPPER = {
    'categorical':'__USE_FEATURE_CATEGORICAL',
    'numerical':'__USE_FEATURE_NUMERICAL',
    'layer':'__LAYER',
    'embbeding_size': '__EMBEDDING_SIZE'
}

SHERPA_ALGORITHM_MAPPER = {
    'bayes':sherpa.algorithms.GPyOpt,
    'random':sherpa.algorithms.RandomSearch,
    'grid_search':sherpa.algorithms.GridSearch,
    'ash':sherpa.algorithms.SuccessiveHalving,
    'local_search':sherpa.algorithms.LocalSearch,
    'pbt':sherpa.algorithms.PopulationBasedTraining,
}


def dict_retrieve(global_dict, key):
    '''
    Retrieves values from module mappers (dicts)
    :param global_dict:
    :param key:
    :return:
    '''
    try: value = global_dict[key]
    except KeyError: raise KeyError(f'Key should be one of {list(global_dict)} not {key}')
    return value

def parameter_mask_setter(parameters, parameter_class):
    '''
    creates a optimizable mask for a given parameter
    :param parameters:
    :param parameter_class:
    :return:
    '''
    type_sufix = parameter_class_sufix_mapper(parameter_class)
    return {parameter:sherpa.Choice(parameter+type_sufix, [True,False]) for parameter in parameters}

def parameter_dict_setter(parameters_dict, parameter_class, sherpa_parameter_type):
    '''
    Creates an optimizable dict of a given parameter type
    :param parameters_dict:
    :param parameter_class:
    :param sherpa_parameter_type: str or mapping
    :return:
    '''
    if sherpa_parameter_type.__class__ == str:
        sherpa_parameter_type = {i:sherpa_parameter_type for i in parameters_dict}

    sherpa_param_class = {k:dict_retrieve(SHERPA_PARAMETER_TYPE_MAPPER, v) for k,v in sherpa_parameter_type.items()}
    type_sufix = parameter_class_sufix_mapper(parameter_class)
    return {name: sherpa_param_class[name](name + type_sufix, value) for name,value in parameters_dict.items()}

def parameter_class_sufix_mapper(parameter_class):
    '''
    Maps parameter type to dictionary key sufix
    :param parameter_class:
    :return:
    '''
    # all value sshould start with and have only one incidence of double underscore "__"
    assert all([not v.split('__')[0] for k,v in CLASS_SUFIX_MAPPER.items()])
    type_sufix = dict_retrieve(CLASS_SUFIX_MAPPER, parameter_class)
    return type_sufix

def parameter_mask_parser(params, parameter_class):
    '''
    Filters params from sherpa.Choice list of params
    :param params: list of sherpa params. each one represents one feature and its respective
    boolean value (keep or dismiss)
    :return: list of parameters kept after applying mask
    '''
    type_sufix = parameter_class_sufix_mapper(parameter_class)
    valid_keys = [i for i in params if i.split('__')[-1] == type_sufix.split('__')[-1]]
    features = [k.split(type_sufix)[0] for k in valid_keys if params[k] == True]
    return features

def parameter_dict_parser(params, parameter_class):
    '''
    Parses params from sherpa.Choice list of params
    :param params:
    :param parameter_class:
    :return: dict containing parameters and its values
    '''
    type_sufix = parameter_class_sufix_mapper(parameter_class)
    valid_keys = [i for i in params if i.split('__')[-1] == type_sufix.split('__')[-1]]
    parameter_dict = {i.split(type_sufix)[0]:params[i] for i in valid_keys}
    return parameter_dict

def parameter_dict_masker(params, dict_parameter_class, mask_parameter_class):
    '''
    Masks a dict of parameters of parameter_class
    :param params:
    :param dict_parameter_class:
    :param mask_parameter_class:
    :return:
    '''
    parameter_dict = parameter_dict_parser(params, dict_parameter_class)
    parameter_msk = parameter_mask_parser(params, mask_parameter_class)
    masked_parameter_dict = {i:parameter_dict[i] for i in parameter_dict if i in parameter_msk}
    return masked_parameter_dict

def create_study(algorithm,parameters,lower_is_better = False, algo_params = {}):
    #handle algorithm input
    if algorithm.__class__ == str:
        algorithm = dict_retrieve(SHERPA_ALGORITHM_MAPPER,algorithm)
    elif isinstance(algorithm,sherpa.algorithms.core.Algorithm):
        #keep value
        pass
    else:
        raise ValueError('algorithm must be of type sherpa.algorithms.core.Algorithm or Str (bayes or random)')
    print(parameters)
    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm(**algo_params),
        lower_is_better=lower_is_better,
        disable_dashboard=True #dashboard disables since it doesnt work on windows
    )
    return study


def parameter_wrapper(name, range, type,scale = 'linear'):
    sherpa_class = dict_retrieve(SHERPA_PARAMETER_TYPE_MAPPER,type)
    if type != 'choice':
        parameter = sherpa_class(name = name, range = range)
    else:
        parameter = sherpa_class(name=name, range=range, scale=scale)
    return parameter
###################### high level fastai part ###############################

def categorical_embbedings_setting(data, categorical_features, max_sz = 300):
    '''
    Sets size of
    :param data:
    :param categorical_features:
    :param max_sz:
    :return:
    '''

    embeddings_sherpa = {}
    for feature in categorical_features:
        cardinality = data[feature].nunique()
        max_emb_sz = min(max_sz, round(1.6 * cardinality ** 0.56))
        min_emb_sz = 1
        type_sufix = parameter_class_sufix_mapper('embbeding_size')
        embeddings_sherpa[feature] = (sherpa.Discrete(feature+type_sufix, [min_emb_sz, max_emb_sz]))
    return embeddings_sherpa

def fastai_layer_parser(n_layers,max_layer_size,layer_shrinkage_factor):
    layers_setup = [max(1,round(max_layer_size*(layer_shrinkage_factor**i))) for i in range(n_layers)]
    return layers_setup

def fastai_sherpa_features_generator(
        data,
        discrete_param_bounds,
        continuous_param_bounds,
        choice_param_bounds,
        categorical_features,
        numeric_features,
        layer_params
):
    '''
    Parses nested params to list of sherpa.core.Parameter
    :param data:
    :param discrete_param_bounds:
    :param continuous_param_bounds:
    :param choice_param_bounds:
    :param categorical_features:
    :param numeric_features:
    :param layer_params:
    :return: list of sherpa parameters
    '''

    #feature selection part
    sherpa_numeric_features = parameter_mask_setter(numeric_features, parameter_class='numerical')
    sherpa_categorical_features = parameter_mask_setter(categorical_features, parameter_class='categorical')
    #embedding size part
    emb_szs = categorical_embbedings_setting(data,categorical_features)
    #layer setup part
    layer_setup = parameter_dict_setter(layer_params,'layer',
                                        {'n_layers':'discrete',
                                         'max_layer_size':'discrete',
                                         'layer_shrinkage_factor':'continuous'})

    #populate parameters list
    opt_parameters = []
    #feature selection features setting
    for param, sherpa_obj in sherpa_numeric_features.items():
        opt_parameters.append(sherpa_obj)
    for param, sherpa_obj in sherpa_categorical_features.items():
        opt_parameters.append(sherpa_obj)
    #embeddings sizes
    for param, sherpa_obj in emb_szs.items():
        opt_parameters.append(sherpa_obj)
    #layers
    for param, sherpa_obj in layer_setup.items():
        opt_parameters.append(sherpa_obj)
    #non nested params
    for param in discrete_param_bounds:
        opt_parameters.append(parameter_wrapper(
            type = 'discrete', **param))
    for param in continuous_param_bounds:
        opt_parameters.append(opt_parameters.append(parameter_wrapper(
            type = 'continuous', **param)))
    for param in choice_param_bounds:
        opt_parameters.append(opt_parameters.append(parameter_wrapper(
            type = 'choice',**param)))

    #removes Nones
    opt_parameters = [i for i in opt_parameters if not i is None]
    return opt_parameters
############################ run study ###################################
def run_study(
        max_iter = 10,
        pipeline_class = None,
        data = None,
        algorithm = None,
        static_params = {},
        study = None,
        discrete_param_bounds = {},
        continuous_param_bounds = {},
        choice_param_bounds = {},
        categorical_features_selection = {},
        numeric_features_selection = {},
        layer_setup = {},
        optimizer_params = {},
        lower_is_better = False,

):
    if study is None:
        #optimizable parameters
        optimizable_parameters = fastai_sherpa_features_generator(
            data,
            discrete_param_bounds,
            continuous_param_bounds,
            choice_param_bounds,
            categorical_features_selection,
            numeric_features_selection,
            layer_setup
        )
        study = create_study(
            algorithm,
            optimizable_parameters,
            lower_is_better=lower_is_better,
            algo_params = optimizer_params
        )
    else:
        study = study

    i = 0

    for trial in tqdm(study, total = max_iter):
        #non nested (float) params
        opt_params = {i: trial.parameters[i] for i in trial.parameters if not ('__' in i)}
        #embedding sizes
        cat_emb_szs = parameter_dict_parser(trial.parameters, parameter_class = 'embbeding_size')
        # feature selection
        categorical_features = parameter_mask_parser(trial.parameters, parameter_class = 'categorical')
        numerical_features = parameter_mask_parser(trial.parameters, parameter_class  = 'numerical')
        #layer setup
        layer_setup_dict = parameter_dict_parser(trial.parameters, parameter_class = 'layer')
        if layer_setup_dict:
            layer_setup = fastai_layer_parser(**layer_setup_dict)
        #all params parsed (filtered by existence)
        pipeline_params_opt = dict(
            cat_emb_szs=cat_emb_szs,
            num_features=numerical_features,
            cat_features=categorical_features,
            fastai_layers_setup=layer_setup,
            **opt_params,
        )
        pipeline_params_opt = {k:v for k,v in pipeline_params_opt.items() if v}
        pipeline_params = {**pipeline_params_opt,**static_params}
        # Create model
        pipeline = pipeline_class(**pipeline_params)

        # Train model
        pipeline.fit(data = data)
        loss, metric = pipeline.learner.validate()
        loss, metric = float(loss), float(metric)
        study.add_observation(trial=trial, iteration=i,
                              objective=metric,
                              context={'loss': loss})

        if i >= max_iter: #study.should_trial_stop(trial):
            break
        i+=1

    #study.finalize(trial=trial) -> kept unfinizhed for further exploration
    return study
