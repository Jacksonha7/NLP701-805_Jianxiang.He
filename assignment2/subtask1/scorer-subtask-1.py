import logging.handlers
import argparse
import sys
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn

"""
Scorer for SemEval 2027 subtask 1.
Since it is a multilabel multiclass classification problem, we compute the micro F1 over all the classes.
One row of the prediction file is the following:
article_id|TAB|begin_span|TAB|end_span|TAB|frame1|TAB|frame2|TAB|...|TAB|frameN

where article_id is the numerical code in the name of the file with the input article, |TAB| is the tab character,
begin_span is the beginning of the span of the paragraph, end span is the end span of the paragraph and frame1|TAB|frame2|TAB|...|TAB|frameN is a tab-separated list of frames (the list could be empty).
For example:

article_id_1	0	56	Quality_of_life
article_id_1	62	381	Quality_of_life
article_id_2	387	780	Policy_prescription_and_evaluation	Quality_of_life
article_id_2	784	1113	Capacity_and_resources
article_id_2	1117	1595	Economic	Fairness_and_equality
article_id_2	1599	2018	Economic	Quality_of_life
article_id_2	2022	2349	Economic	Fairness_and_equality

An empty frame list (no framing detected) is scored against an explicit "None" class.
"""

NONE_LABEL = "None"

logger = logging.getLogger("subtask1_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def read_frame_list_from_file(file_full_name):
  """
  Read the list of frame names from a file, one per line.
  """
  with open(file_full_name, encoding='utf-8') as f:
    return [ line.rstrip() for line in f.readlines() ]


def _read_csv_input_file(file_full_name):
  """
  Read a csv file with two columns TAB separated:
   - first column is the id of the example
   - second column is the begin span
   - third column is the end span
   - remaining columns is the list of labels of the example
  """
  d = list()
  with open(file_full_name, encoding='utf-8') as f:
    for line in f.readlines():
      line = line.rstrip('\n')
      if not line:
        continue
      line_data = line.split("\t")
      if len(line_data) < 3:
        logger.error('ERROR: each line of the file is supposed to have at least a doc_id, a begin span, and an end span, found inconsistencies in line ' + line)
        sys.exit(1)
      a = dict()
      a["doc_id"] = line_data.pop(0)
      a["begin_span"] = line_data.pop(0)
      a["end_span"] = line_data.pop(0)
      a["frames"] = line_data
      d.append(a)
  return d


def _labels_correct(labels, CLASSES, debug=False):
  """
  Make sure all the labels correspond to strings in the CLASSES array
  :param labels: a list of dictionaries
  :param CLASSES: a list of allowed labels
  """
  if debug:
    s=""
    for entry in labels:
     for l in entry["frames"]:
         if l not in CLASSES:
             s += l + " "
    return s
  else:
    for entry in labels:
      for l in entry["frames"]:
        if l not in CLASSES:
          return False
  return True


def _key(entry):
  """
  Unique key for a row: doc_id and span.
  """
  return entry["doc_id"]+"_"+entry["begin_span"]+"_"+entry["end_span"]


def _dedupe_keeping_last(labels, source_name):
  """
  Keep the last occurrence of each duplicate row, warning about the rest.
  """
  deduped = {}
  for entry in labels:
    key = _key(entry)
    if key in deduped:
      logger.warning('WARNING: duplicate entry for "%s" found in the %s file; only the last occurrence will be used for scoring.'
                     % (key, source_name))
    deduped[key] = entry
  return list(deduped.values())


def _correct_number_of_examples(pred_labels, gold_labels):
  """
  Make sure that the number of predictions is exactly the same as the gold labels
  """
  return len(pred_labels)==len(gold_labels)


def _correct_id_list(pred_labels, gold_labels, debug=False):
  """
  Check that the keys of pred_labels match the gold file.
  """
  pred_ids = set(_key(entry) for entry in pred_labels)
  gold_ids = set(_key(entry) for entry in gold_labels)
  if debug:
    return ", ".join(pred_ids.symmetric_difference(gold_ids))
  return pred_ids == gold_ids


def _extract_matching_lists(pred_labels, gold_labels):
  """
  Extract the list of values from the two list of dictionaries ensuring that elements with the same key are in the same position.
  """
  pred_by_id = {_key(entry): entry["frames"] for entry in pred_labels}
  pred_values, gold_values = ([],[])
  for entry in gold_labels:
    gold_values.append(entry["frames"])
    pred_values.append(pred_by_id[_key(entry)])
  return pred_values, gold_values


def _to_scoring_labels(frame_lists):
  """
  Map an empty frame list to the None label.
  """
  return [labels if labels else [NONE_LABEL] for labels in frame_lists]


def correct_format(pred_labels, gold_labels, CLASSES):
  """
  Check whether the format of the prediction file is correct.
  The number of checks that can be performed depends on the availability of the gold labels.
  Runs every check and reports all problems found, rather than stopping at the first one.
  """
  is_correct = True

  if not _labels_correct(pred_labels, CLASSES):
    logger.error('ERROR: the following labels in the prediction file are not valid: {}.'
                 .format(_labels_correct(pred_labels, CLASSES, True)))
    is_correct = False

  if gold_labels: # we can do further checks if the gold_labels are available
    if not _correct_number_of_examples(pred_labels, gold_labels):
      logger.error('The number of predictions (%d) is not the expected one (%d)'
                   %(len(pred_labels), len(gold_labels)))
      is_correct = False

    if not _correct_id_list(pred_labels, gold_labels):
      logger.error('The list of articles ids+spans is not correct. The following entries are not in the gold file or absent from the prediction file: %s'
                   %(_correct_id_list(pred_labels, gold_labels, True)))
      is_correct = False

  return is_correct


def evaluate(pred_labels, gold_labels, CLASSES):
  """
    Evaluates the predicted classes w.r.t. a gold file.
    Metrics are: multilabel macro_f1 and micro_f1.
    :param pred_labels: a dictionary with predictions,
    :param gold_labels: a dictionary with gold labels.
  """
  pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels)
  pred_values = _to_scoring_labels(pred_values)
  gold_values = _to_scoring_labels(gold_values)

  mlb = MultiLabelBinarizer()
  mlb.fit([CLASSES + [NONE_LABEL]])
  gold_values = mlb.transform(gold_values)
  pred_values = mlb.transform(pred_values)

  macro_f1 = f1_score(gold_values, pred_values, average="macro", zero_division=1)
  micro_f1 = f1_score(gold_values, pred_values, average="micro", zero_division=1)
  return macro_f1, micro_f1


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_file_path", '-g', type=str, required=False, help="Paths to the file with gold annotations.")
  parser.add_argument("--pred_file_path", '-p', type=str, required=True, help="Path to the file with predictions")
  parser.add_argument("--frame_file_path", '-f', type=str, required=True, help="Path to the file with the names of the frames")
  parser.add_argument("--log_to_file", "-l", action='store_true', default=False,
                      help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log")
  parser.add_argument('--output-for-script', "-o", dest='output_for_script', required=False, action='store_true',
                      default=False, help="Prints the output in a format easy to parse for a script")
  args = parser.parse_args()

  output_for_script = bool(args.output_for_script)
  if not output_for_script:
    logger.addHandler(ch)

  CLASSES = read_frame_list_from_file(args.frame_file_path)

  pred_file = args.pred_file_path
  if args.gold_file_path:
    gold_file = args.gold_file_path
  else:
    gold_file = None
  if args.log_to_file:
    output_log_file = pred_file + ".log"
    logger.info("Logging execution to file " + output_log_file)
    fileLogger = logging.FileHandler(output_log_file)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(formatter)
    logger.addHandler(fileLogger)
    logger.setLevel(logging.DEBUG) #

  if args.log_to_file:
    logger.info('Reading predictions file')
  else:
    logger.info('Reading predictions file {}'.format(args.pred_file_path))
  if gold_file:
    if args.log_to_file:
      logger.info('Reading gold file')
    else:
      logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
  else:
    logger.info('No gold file provided')

  pred_labels = _read_csv_input_file(pred_file)
  gold_labels = _read_csv_input_file(gold_file) if gold_file else None

  pred_labels = _dedupe_keeping_last(pred_labels, "predictions")
  if gold_labels:
    gold_labels = _dedupe_keeping_last(gold_labels, "gold")

  if correct_format(pred_labels, gold_labels, CLASSES):
    logger.info('Prediction file format is correct')
    if gold_labels:
      macro_f1, micro_f1 = evaluate(pred_labels, gold_labels, CLASSES)
      logger.info("micro-F1={:.5f}\tmacro-F1={:.5f}".format(micro_f1, macro_f1))
      if output_for_script:
        print("{}\t{}".format(micro_f1, macro_f1))
