import json
import logging

def write_hparams(hparams, filename):
  """Writes HParams to disk as JSON.

  Args:
    hparams: HParams object or dict.
    filename: String output filename.
  """
  with open(filename, 'w') as f:
    if hasattr(hparams, 'to_json'):
      # 如果 hparams 有 to_json 方法
      f.write(hparams.to_json(indent=2, sort_keys=True, separators=(',', ': ')))
    else:
      # 如果 hparams 是字典或其他对象，转换为 JSON
      json.dump(hparams.__dict__ if hasattr(hparams, '__dict__') else hparams, 
                f, indent=2, sort_keys=True, separators=(',', ': '))


def read_hparams(filename, defaults):
  """Reads HParams from JSON.

  Args:
    filename: String filename.
    defaults: HParams containing default values.

  Returns:
    HParams.

  Raises:
    FileNotFoundError: If the file cannot be read.
    ValueError: If the JSON record cannot be parsed.
  """
  with open(filename) as f:
    logging.info('Reading HParams from %s', filename)
    return defaults.parse_json(f.read())