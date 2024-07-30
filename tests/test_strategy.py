import pytest
from wrtrade.strategy import Strategy

def test_strategy_abstract_class():
    with pytest.raises(TypeError):
        Strategy()  # Should raise TypeError as it's an abstract class