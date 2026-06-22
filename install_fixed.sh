#/bin/bash

# fixed install script from MLRC Bench (fixes existing issues)
pip install -r requirements.txt
pip install typing-inspect==0.8.0 typing-extensions>=4.15.0
pip install pydantic -U
pip install -U numpy>=1.26.4,<2.0
pip install --force-reinstall charset-normalizer==3.1.0