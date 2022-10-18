import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

log.info("Loading package")
from src.root_dataloader import ROOTCLoud

log.info("Loading data")
ds = ROOTCLoud()
print(ds[0])