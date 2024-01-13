import editdistance
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 0 if len(predicted_text) == 0 else 1
    return editdistance.eval(predicted_text, target_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    return calc_cer(target_text.split(' '), predicted_text.split(' '))