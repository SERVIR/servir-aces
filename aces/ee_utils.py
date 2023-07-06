# -*- coding: utf-8 -*-

import ee
from typing import Union

__all__ = ["EEUtils"]

class EEUtils:
    """
    EEUtils: Earth Engine Utility Class

    This class provides utility functions to handle Earth Engine API information and make authenticated requests.
    """
    @staticmethod
    def get_credentials_by_service_account_key(key):
        """
        Helper function to retrieve credentials using a service account key.

        Parameters:
        key (str): The path to the service account key JSON file.

        Returns:
        ee.ServiceAccountCredentials: The authenticated credentials.
        """
        import json
        service_account = json.load(open(key))
        credentials = ee.ServiceAccountCredentials(service_account['client_email'], key)
        return credentials

    @staticmethod
    def initialize_session(use_highvolume : bool = False, key : Union[str, None] = None):
        """
        Initialize the Earth Engine session.

        Parameters:
        use_highvolume (bool): Whether to use the high-volume Earth Engine API.
        key (str or None): The path to the service account key JSON file. If None, the default credentials will be used.
        """
        if key is None:
            if use_highvolume:
                ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
            else:
                ee.Initialize()
        else:
            credentials = EEUtils.get_credentials_by_service_account_key(key)
            if use_highvolume:
                ee.Initialize(credentials, opt_url="https://earthengine-highvolume.googleapis.com")
            else:
                ee.Initialize(credentials)
