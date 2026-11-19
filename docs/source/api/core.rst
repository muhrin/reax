Core API
========

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class.rst

   reax.Module
   reax.Engine

Checkpointing classes
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class.rst

   reax.Checkpointing
   reax.MsgpackCheckpointing

Saving helpers
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   reax.save_hparams_to_yaml
   reax.load_hparams_from_yaml

Randomness
----------

See :doc:`/user_guide/randomness`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   reax.random.create_rngs
   reax.random.to_key
   reax.random.draw_seed
   reax.random.seed_globals
   reax.random.rngs_state_dict
   reax.random.load_rngs_state_dict
   reax.seed_everything
