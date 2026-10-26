class ServiceUnavailable(Exception):
    """An external API failed (timeout, 5xx, bad response). A cached fallback may be used."""


class NoDataError(Exception):
    """The API worked but has no data for this spot (e.g. open water). Never use a fallback."""