Code Graphs
===========

This is a module to hold functions for creating graphs that document the code.



Module Diagram
--------------

This will create class diagram for classes found in a module (see the :ref:`Exploring Pyreverse <exploring-pyreverse-module>` section). Since it is using `pyreverse` it has to be installed and available on the execution path.

.. currentmodule:: arachneape.commoncode.code_graphs
.. autosummary::
   :toctree: api

   module_diagram



Class Diagram
-------------

This creates a more detailed class diagram. Unlike the module-diagram, this requires a specific class name (because it only shows one class). See :ref:`Exploring Pyreverse <exploring-pyreverse-class-diagram>` for more detail.

.. autosummary::
   :toctree: api

   class_diagram

