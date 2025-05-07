"""
This module provides class abstraction for configure and show elements in streamlit.
"""
import os
import abc

DEFAULT_DIR = os.path.join(".", "aux")

class StElement(abc.ABC):
    """ Abstract class to represent tipical streamlit flux """

    @abc.abstractmethod
    def configure(self):
        pass

    @abc.abstractmethod
    def check_dialogs(self):
        pass

    @abc.abstractmethod
    def plot(self):
        pass
