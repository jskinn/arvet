from pymodm.connection import connect as pymodm_connect


def configure(config: dict, override_host: str = None, override_port: int = None) -> None:
    """
    Connect pymodm to the database
    configuration can be either a single key, 'uri', specifying the full connection URI.
    Otherwise, you can provide 'host', 'port', and 'database' keys and we'll build the URI for you.

    :param config: A dict containing configuration parameters. See arvet.config.global_configuration
    :param override_host: Override the configured database host
    :param override_port: Override the configured database port
    :return: None
    """
    if 'database' in config:
        config = config['database']
    if 'uri' in config and override_host is None and override_port is None:
        mongodb_uri = config['uri']
    else:
        # handle overridden values for the host and port
        if override_host is None:
            override_host = config.get('host', 'localhost')
        if override_port is None:
            override_port = config.get('port', '27017')

        mongodb_uri = "mongodb://{host}:{port}/{database}".format(
            host=override_host,
            port=override_port,
            database=config.get('database', 'arvet-db')
        )
    pymodm_connect(mongodb_uri)
