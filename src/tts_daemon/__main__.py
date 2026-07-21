"""Allow ``python -m tts_daemon`` as an alternative to the console script."""

import sys

from tts_daemon.cli import main

if __name__ == "__main__":
    sys.exit(main())
