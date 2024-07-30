from abc import ABC, abstractmethod
from pandas import Series, DataFrame

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    @abstractmethod
    def generate_signals(self, market_data: DataFrame) -> Series:
        """
        Generate trading signals based on the given data.

        :param data: Market data
        :return: Pandas Series of trading signals
        """
        pass 