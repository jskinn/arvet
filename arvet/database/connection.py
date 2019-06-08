from pymodm.connection import connect as pymodm_connect


def configure(config: dict) -> None:
    """
    Connect pymodm to the database
    configuration can be either a single key, 'uri', specifying the full connection URI.
    Otherwise, you can provide 'host', 'port', and 'database' keys and we'll build the URI for you.

    :param config: A dict containing configuration parameters. See arvet.config.global_configuration
    :return: None
    """
    if 'database' in config:
        config = config['database']
    if 'uri' in config:
        mongodb_uri = config['uri']
    else:
        mongodb_uri = "mongodb://{host}:{port}/{database}".format(
            host=config.get('host', 'localhost'),
            port=config.get('port', '27017'),
            database=config.get('database', 'arvet-db')
        )
    pymodm_connect(mongodb_uri)
