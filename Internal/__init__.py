from typing import *

def ImportingThrow(
    ex:             ImportError,
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = "{module} Module requires {required} package.",
    installBase:    str = "\tpip install {name}"
    ):
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(messageBase.format_map(dict(module=moduleName, required=requierds_str)))
        print('Install it via command:')
        for i in requierds:
            print(installBase.format_map(name=i))
        if ex:
            raise ex

def InternalImportingThrow(
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = "{module} Module requires internal lekit package: {required}.",
    ):
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(f"Internal lekit package is not installed.\n{messageBase.format_map(dict(module=moduleName, required=requierds_str))}")