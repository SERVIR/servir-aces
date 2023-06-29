# -*- coding: utf-8 -*-

import ee
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession


# this part was mostly inspired from https://github.com/KMarkert/restee/blob/main/restee/core.py and adapted to our needs
class EESession:
    """EESession class that handles EE API info to make authenticated requests.
    """
    def __init__(self, project: str, service_account_key: str, use_service_account: bool = False):
        """Initialization function for the EESession class

        args:
            project (str): Google Cloud project name with service account whitelisted to use Earth Engine
            key (str): path to private key file for your whitelisted service account
        """
        self._PROJECT = project
        self._SESSION = self._get_session_by_service_account_key(service_account_key) if use_service_account else self._get_session()

    @property
    def cloud_project(self):
        return self._PROJECT

    @property
    def session(self):
        return self._SESSION

    @staticmethod
    def _get_session_by_service_account_key(key):
        """Helper function to authenticate"""
        import json
        service_account = json.load(open(key))
        credentials = ee.ServiceAccountCredentials(service_account['client_email'], key)
        return credentials

    @staticmethod
    def _get_session():
        return ee.Authenticate()
