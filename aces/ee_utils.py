# -*- coding: utf-8 -*-

import ee


class EEUtils:
    """EESession class that handles EE API info to make authenticated requests.
    """
    @staticmethod
    def get_credentials_by_service_account_key(key):
        """Helper function to authenticate"""
        import json
        service_account = json.load(open(key))
        credentials = ee.ServiceAccountCredentials(service_account['client_email'], key)
        return credentials

    @staticmethod
    def initialize_session(use_highvolume : bool = False, key : str | None = None):
        """Initialize EE session"""
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

