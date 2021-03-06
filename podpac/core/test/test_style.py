
from __future__ import division, unicode_literals, print_function, absolute_import

from podpac.core.node import Node
from podpac.core.style import Style

class TestStyleCreation(object):
    def test_basic_creation(self):
        s = Style()

    def test_create_with_node(self):
        node = Node()
        s = Style(node)

    def test_get_default_cmap(self):
        style = Style()
        style.cmap