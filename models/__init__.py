import logging
logger = logging.getLogger('base')


def create_model(opt):
    # image restoration
    from .SIEN_model import SIEN_Model as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
