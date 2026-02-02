import yaml
import re

# https://gist.github.com/pbsds/f22ab3596977b43c935171e53f226cd9

class MyYamlLoader(yaml.Loader):
  pass

MyYamlLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=MyYamlLoader)