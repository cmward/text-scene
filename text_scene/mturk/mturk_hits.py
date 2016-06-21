import csv
import sys
import os
from collections import defaultdict, Counter
from boto.mturk.connection import MTurkConnection, MTurkRequestError
from boto.mturk.question import QuestionContent, Question, QuestionForm
from boto.mturk.question import Overview, AnswerSpecification
from boto.mturk.question import SelectionAnswer, FormattedContent
from boto.mturk.question import FreeTextAnswer, Binary
from boto.mturk.qualification import NumberHitsApprovedRequirement
from boto.mturk.qualification import PercentAssignmentsApprovedRequirement
from boto.mturk.qualification import Qualifications
from paths import (
    INSTRUCTIONS_HTML,
    SENTENCES_CSV,
    IMG_URLS,
    REJECTED_IMGS_FILE,
    ANNOTATED_IMGS_FILE,
    KEY_FILE,
    MTURK_RESULTS_CSV,
    GOLD_MTURK_RESULTS_CSV,
    ALL_HITS_CSV
)

"""
Adapted from http://www.toforge.com/2011/04/boto-mturk-tutorial-create-hits/
"""


with open(KEY_FILE) as f:
    lines = f.readlines()
keys = [x.strip().split('=') for x in lines]
ACCESS_ID = keys[0][1]
SECRET_KEY = keys[1][1]
SB_HOST = 'mechanicalturk.sandbox.amazonaws.com'
REAL_HOST = 'mechanicalturk.amazonaws.com'

mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=SB_HOST)

def make_hit(image_url):
    title = 'Label image with its location'
    description = 'Answer questions about an image to label its location.'
    keywords = 'image categorization, locations, scene recognition'

    in_out = [('indoors','0'), ('outdoors','1')]
    nat_manmade = [('man-made','0'), ('natural','1')]
    functions = [('transportation/urban','0'),
                 ('restaurant','1'),
                 ('recreation','2'),
                 ('domestic','3'),
                 ('work/education','4'),
                 ('other/unclear','5')]
    landscapes = [('body of water/beach','0'),
                  ('field','1'),
                  ('mountain','2'),
                  ('forest/jungle','3'),
                  ('other/unclear','4')]

    #---------------  BUILD OVERVIEW -------------------

    overview = Overview()
    overview.append_field('Title', title)
    with open(INSTRUCTIONS_HTML) as html:
        instructions = html.read()
    overview.append(FormattedContent(instructions))

    image = Binary('image', None, image_url, 'image')
    overview.append(image)

    #---------------  BUILD QUESTION 1 -------------------

    qc1 = QuestionContent()
    qc1.append_field('Text',
                     'Is the location shown in the image indoors or outdoors?')

    fta1 = SelectionAnswer(min=1, max=1,style='checkbox',
                          selections=in_out,
                          type='text',
                          other=False)

    q1 = Question(identifier='Question 1',
                  content=qc1,
                  answer_spec=AnswerSpecification(fta1),
                  is_required=True)

    #---------------  BUILD QUESTION 2 -------------------

    qc2 = QuestionContent()
    qc2.append_field('Text',
                     'Is the location shown in the image man-made or ' +
                     'natural? Examples of man-made locations include ' +
                     'buildings and parks while examples of natural ' +
                     'locations include mountains and rivers.')

    fta2 = SelectionAnswer(min=1, max=1,style='checkbox',
                          selections=nat_manmade,
                          type='text',
                          other=False)

    q2 = Question(identifier='Question 2',
                  content=qc2,
                  answer_spec=AnswerSpecification(fta2),
                  is_required=True)

    #---------------  BUILD QUESTION 3 -------------------

    qc3 = QuestionContent()
    qc3.append_field('Text',
                     'If the location in the image is man-made, what is the ' +
                     'general function or type of the location? If the ' +
                     'location is natural (not man-made), don\'t select ' +
                     'anything here.')

    fta3 = SelectionAnswer(min=0, max=1,style='checkbox',
                          selections=functions,
                          type='text',
                          other=False)

    q3 = Question(identifier='Question 3',
                  content=qc3,
                  answer_spec=AnswerSpecification(fta3),
                  is_required=False)

    #---------------  BUILD QUESTION 4 -------------------

    qc4 = QuestionContent()
    qc4.append_field('Text',
                     'If the location in the picture is natural, what ' +
                     'kind of natural location is it? If the location ' +
                     'man-made (not natural), don\'t select anything here.')

    fta4 = SelectionAnswer(min=0, max=1,style='checkbox',
                          selections=landscapes,
                          type='text',
                          other=False)

    q4 = Question(identifier='Question 4',
                  content=qc4,
                  answer_spec=AnswerSpecification(fta4),
                  is_required=False)

    #--------------- BUILD THE QUESTION FORM -------------------

    question_form = QuestionForm()
    question_form.append(overview)
    question_form.append(q1)
    question_form.append(q2)
    question_form.append(q3)
    question_form.append(q4)

    #-------------- QUALIFICATIONS -------------------

    percent = PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo', 95)
    number = NumberHitsApprovedRequirement('GreaterThanOrEqualTo', 200)
    quals = Qualifications()
    quals.add(percent)
    quals.add(number)

    #--------------- CREATE THE HIT -------------------

    mtc.create_hit(questions=question_form,
                   max_assignments=1,
                   title=title,
                   description=description,
                   keywords=keywords,
                   qualifications=quals,
                   annotation=image_url,
                   duration=60*10,
                   reward=0.03)

def make_hit_batch(log_file, n_images=100, redo=False, redo_log=None,
                   img_url_file=IMG_URLS):
    """
    Parameters
    ----------
    log_file: path to file listing which images have already been annotated
    n_images: how many images to make HITs for
    redo: if True, make HITs for images in `redo_log`
    redo_log: path to file specifying which images to make HITs for
        (rejected/no majority)
    img_url_file: path to file containing all image urls
    """
    all_images = set()
    if redo:
        not_annotated = set()
        with open(redo_log, 'r') as log:
            for line in log:
                not_annotated.add(line.strip())
        not_annotated = list(not_annotated)
        n_images = len(not_annotated)
    else:
        with open(img_url_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                all_images.add(row[0])
        annotated = set()
        with open(log_file, 'r') as log:
            for line in log:
                annotated.add(line.strip())
        not_annotated = list(all_images - annotated)
    with open(log_file, 'a') as log:
        for i in range(n_images):
            make_hit(not_annotated[i])
            log.write(not_annotated[i]+'\n')

def get_all_reviewable_hits(mtc):
    page_size = 50
    hits = mtc.get_reviewable_hits(page_size=page_size)
    print 'Total results to fetch: {}'.format(hits.TotalNumResults)
    print 'Request hits page: 1'
    total_pages = float(hits.TotalNumResults) / page_size
    int_total = int(total_pages)
    if total_pages - int_total > 0:
        total_pages = int_total + 1
    else:
        total_pages = int_total
    pn = 1
    while pn < total_pages:
        pn = pn + 1
        print 'Request hits page: {}'.format(pn)
        temp_hits = mtc.get_reviewable_hits(page_size=page_size,
                                            page_number=pn)
        hits.extend(temp_hits)
    return hits

def approve_and_pay_all(hits, outfile, log_file, check_valid=True):
    """
    Approves/reject all available hits, write results to `outfile`,
    write image files with rejected hits to `log_file`.
    """
    # row:
    #   0: requester annotation
    #   1: worker id
    #   2: q1
    #   3: q2
    #   4: q3
    #   5: q4
    rejected_assignments = defaultdict(list)
    rejected_workers = Counter()
    with open(outfile, "ab") as f, open(ALL_HITS_CSV, "ab") as allhits:
        writer = csv.writer(f, lineterminator='\n')
        allhits_writer = csv.writer(allhits)
        for hit in hits:
            h = mtc.get_hit(hit.HITId)[0]
            if h.HITReviewStatus == 'Reviewing': continue
            ra = h.RequesterAnnotation
            assignments = mtc.get_assignments(hit.HITId)
            for assignment in assignments:
                try:
                    row = [ra]
                    row.append(assignment.WorkerId)
                    rejected = False
                    for question_form_answer in assignment.answers[0]:
                        if not question_form_answer.fields:
                            row.append('NA')
                        for answer in question_form_answer.fields:
                            row.append(answer)
                    if check_valid:
                        reject_msg = check_row(row) # returns msg if rejected
                        if reject_msg:
                            rejected = True
                            mtc.reject_assignment(assignment.AssignmentId,
                                                  feedback=reject_msg)
                            rejected_assignments[ra].append(reject_msg)
                            rejected_workers[assignment.WorkerId] += 1
                            with open(log_file, 'a') as log:
                                log.write(ra + '\n')
                    allhits_writer.writerow([hit.HITId, assignment.WorkerId] + row)
                    if not rejected:
                        mtc.approve_assignment(assignment.AssignmentId)
                        writer.writerow(row)
                except MTurkRequestError:
                    continue
            mtc.set_reviewing(hit.HITId)
            #mtc.disable_hit(hit.HITId)
    print "Rejected %i assignments:" % len(rejected_assignments)
    for rejected_assignment in rejected_assignments:
        print rejected_assignment
    for worker, count in rejected_workers.items():
        print worker, count

def check_row(row):
    """Returns reject_msg if row is invalid, None otherwise."""
    reject_msg = ""
    # selections provided for landscape and function
    if row[4] != 'NA' and row[5] != 'NA':
        reject_msg = ("Answers specified for both "
                      "natural landscape type (Question 4) and function "
                      "(Question 3). ")
    # indoors and landscape selected
    elif row[2] == '0' and row[5] != 'NA':
        reject_msg = ("Selected indoors for Question 1 and "
                      "a natural landscape type (Question 4).")
    # man-made and landscape selected
    elif row[3] == '0' and row[5] != 'NA':
        reject_msg = ("Selected man-made for Question 2 and "
                      "an answer for Question 4 (landscape type).")
    # natural and function selected
    elif row[3] == '1' and row[4] != 'NA':
        reject_msg = ("Selected an answer for function (Question 3) and "
                      "a natural landscape type (Question 4)")
    elif row[4] == 'NA' and row[5] == 'NA':
        reject_msg = ("Did not select answers for Question 3 and Question 4 "
                      "(one of them needs to be answered)")
    return reject_msg

def disable_all_hits():
    hits = mtc.get_all_hits()
    for hit in hits:
        mtc.expire_hit(hit.HITId)
        mtc.disable_hit(hit.HITId)

def main(make_hits=False, approve=False, redo_hits=False, redo_log=None,
         n_images=100, outfile=None, log_file=None):
    if approve:
        hits = get_all_reviewable_hits(mtc)
        approve_and_pay_all(hits, outfile, log_file, check_valid=True)
    elif make_hits:
        make_hit_batch(log_file, n_images=n_images)
    elif redo_hits:
        make_hit_batch(log_file, n_images=n_images, redo=True,
                       redo_log=redo_log)
    else:
        raise Exception("Didn't get a recognized keyword")
