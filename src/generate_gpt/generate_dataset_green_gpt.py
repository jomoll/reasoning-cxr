import random
import pandas as pd
import itertools
from call_gpt4 import call_gpt4v, call_gpt4_turbo
import json
import re
from tqdm import tqdm
import time
import asyncio
from multiprocessing import Pool
from all_list import get_reports_as_list

# the seed is used to randomize the reports for multiple parallel runs
ROOT = "data/new_chestxray_data"

SEED = random.randint(0, 10000000)
INPUT_PATH = ROOT
column_names = None
OUTPUT_PATH = f"{ROOT}/generated_dataset_seed_{SEED}.json"
NUM_SAMPLES = 795467

generate = False
if SEED is not None:
    print(f"Seed: {SEED}")
    generate = True


class CandiateReport:
    def __init__(self):
        self.reference_reports = None
        self.error_types = [
            "false report of a finding in the candidate",
            "missing a finding present in the reference",
            "misidentification of a finding's anatomic location/position",
            "misassessment of the severity of a finding",
            "mentioning a comparison that isn't in the reference",
            "omitting a comparison detailing a change from a prior study",
        ]
        self.cost = 0

    def get_sentences(self, report):
        return report.split(".")

    def reorder_sentences(self, report):
        sentences = self.get_sentences(report)
        random.shuffle(sentences)
        reordered_report = " ".join(sentences)
        return reordered_report

    def drop_sentences(self, report):
        sentences = self.get_sentences(report)
        number_of_sentences = random.randint(0, len(sentences) - 1)
        dropped_sentences = random.sample(sentences, number_of_sentences)
        for sentence in dropped_sentences:
            report = report.replace(sentence, "")
        return report

    def random_pairing(self, report=None):
        return random.choice(self.reference_reports)

    def get_error_combination(self, report):
        number_of_error_types = random.randint(0, len(self.error_types))
        if number_of_error_types == 0:
            return "no errors"
        number_of_error_types = min(
            number_of_error_types, len(self.get_sentences(report))
        )

        # upsample_last_errorcategories = random.choice([0, 1])
        # if upsample_last_errorcategories:
        #     self.error_types = self.error_types + self.error_types[-2:] * 2

        combinations = list(
            itertools.combinations(self.error_types, number_of_error_types)
        )
        list_of_errors = random.choice(combinations)
        string_of_errors = (
            ", ".join(list_of_errors[:-1]) + " and " + list_of_errors[-1]
            if len(list_of_errors) > 1
            else list_of_errors[0]
        )
        return string_of_errors

    def get_prompt(self, report):
        error_types = self.get_error_combination(report)
        subtle_change = (
            "Aim for subtlety, adjusting only one word where feasible. "
            if random.random() > 0.5
            else ""
        )
        if error_types != "no errors":
            return f"[Objective]: Create a candidate radiology report that subtly integrates specific errors based on the provided reference report. \n Process Overview: You will be presented with:  \n 1. Style of errors.  \n 2.A reference radiology report to base your candidate report on. \n 3. The desired format for your candidate report. Note: Be short in your response! \n\n Style of errors: \n Introduce errors related to {error_types}. The errors should be woven into the report as if they were genuine observations from a medical image, without any meta-commentary on their accuracy. {subtle_change}Be concise! \n Reference Report: \n{report}\n Desired format for your candidate report:  \n\n [Candidate]: <Candidate Report>"
        return f"[Objective]: Create a candidate radiology report that has the same clinical meaning but is slightly rephrased. \n Process Overview: You will be presented with:  \n 1.A reference radiology report to base your candidate report on. \n 2. The desired format for your candidate report. Note: Be short in your response! \n\n Reference radiology report: \n{report}\n\n Desired format for your candidate report:  \n\n [Candidate]: <Candidate Report>"

    async def modified_report(self, report):
        prompt = self.get_prompt(report)
        try:
            cost, completion = await call_gpt4_turbo("", prompt, temperature=0, n=1)
            completion = completion.replace("[Candidate]: ", "")
            self.cost += cost
            print(f"Generated candidate report. Cost: {cost}")
            return completion
        except Exception as e:
            print(f"Error in modified_report: {e}")
            return None

    async def get_candidate_report(self, report, reports):
        self.reference_reports = reports
        types = {
            "reorder": self.reorder_sentences,
            "drop": self.drop_sentences,
            "random": self.random_pairing,
            "modify": self.modified_report,
        }
        frequency = {
            "reorder": 0,
            "drop": 0,
            "random": 0,
            "modify": 1,
        }
        modification = random.choices(
            list(types.keys()), list(frequency.values()), k=1
        )[0]

        try:
            if modification == "modify":
                result = await types[modification](report)
            else:
                result = types[modification](report)

            print(f"Generated candidate report using {modification} method")
            return result, types[modification].__name__
        except Exception as e:
            print(f"Error in get_candidate_report: {e}")
            return None, None


class GREENDataset:
    def __init__(self, path=None, column_names=None):
        self.reference_reports = None
        self.reference = None
        self.candidates = None
        if column_names is not None:
            if len(column_names) == 1:
                print(
                    "Only one column name provided, using the first column name as the reference report and formating it"
                )
                self.reference_reports = self.get_reference_reports(
                    path, column_names[0]
                )
            if len(column_names) == 2:
                print(
                    "Two column names provided, using the first column name as the reference report and the second column name as the candidate report"
                )
                self.eval_reports = pd.read_csv(path)
                self.reference = self.eval_reports[column_names[0]]
                self.candidates = self.eval_reports[column_names[1]]
                print("Number of reference reports:", len(self.reference))
                print("Number of candidate reports:", len(self.candidates))
        else:
            self.reference_reports = get_reports_as_list(ROOT)

        print("Number of reference reports:", len(self.reference_reports))

        self.make_candidate_report = CandiateReport()
        self.cost = 0
        self.elapsed_time = 0
        self.dataset = []

    def process_report(self, report):
        if "IMPRESSION" in report and "FINDINGS" in report:
            split_by = random.choice(["IMPRESSION", "FINDINGS"])
        elif "Impression" in report and "Findings" in report:
            split_by = random.choice(["Impression", "Findings"])
        elif "IMPRESSION" in report:
            split_by = "IMPRESSION"
        elif "Impression" in report:
            split_by = "Impression"
        elif "Findings" in report:
            split_by = "Findings"
        else:
            split_by = "FINDINGS"

        report = report.split(split_by)[1]
        split_pattern = r"\s(?=[A-Z]{2,}\b)"
        report_split = re.split(split_pattern, report)[0]
        if report_split != "":
            report = report_split

        if report.startswith(": "):
            report = report[2:]
        report = re.sub(r"(?<=\w)\n(?=\w)", " ", report)
        report = report.replace("\n", "")

        return report

    def process_reports_in_parallel(self, reports):
        with Pool() as pool:
            processed_reports = list(
                tqdm(pool.imap(self.process_report, reports), total=len(reports))
            )
        return processed_reports

    def get_reference_reports(self, path_reference_reports, column_names="text"):
        reference_reports = pd.read_csv(path_reference_reports)
        reference_reports = reference_reports.sample(100000, random_state=SEED)
        print("Filtering reports...")
        for report in tqdm(reference_reports[column_names]):
            if "IMPRESSION" not in report and "FINDINGS" not in report:
                reference_reports = reference_reports[
                    reference_reports["text"] != report
                ]
        print("Splitting reports...")
        modified_reports = self.process_reports_in_parallel(
            reference_reports[column_names]
        )
        return modified_reports

    async def get_random_sample(self):
        reference = random.choice(self.reference_reports)
        candidate, error_type = await self.make_candidate_report.get_candidate_report(
            reference, self.reference_reports
        )
        return reference, candidate, error_type

    async def make_prompt(self, reference=None, candidate=None, error_type=None):
        if reference is None or candidate is None:
            reference, candidate, error_type = await self.get_random_sample()
        if candidate is None:
            print("Failed to generate candidate report")
            return None, None, None, None

        prompt = f"""Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

    Process Overview: You will be presented with:

    1. The criteria for making a judgment.
    2. The reference radiology report.
    3. The candidate radiology report.
    4. The desired format for your assessment.

    1. Criteria for Judgment:

    For each candidate report, determine:

    The count of clinically significant errors.
    The count of clinically insignificant errors.

    Errors can fall into one of these categories:

    a) False report of a finding in the candidate.
    b) Missing a finding present in the reference.
    c) Misidentification of a finding's anatomic location/position.
    d) Misassessment of the severity of a finding.
    e) Mentioning a comparison that isn't in the reference.
    f) Omitting a comparison detailing a change from a prior study.
    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.

    2. Reference Report:
    {reference}

    3. Candidate Report:
    {candidate}

    4. Reporting Your Assessment:

    Follow this specific format for your output, even if no errors are found:
    ```
    [Explanation]:
    <Explanation>

    [Clinically Significant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

    [Clinically Insignificant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

    [Matched Findings]:
    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
    ```
    """

        return prompt, reference, candidate, error_type

    async def get_response(self, reference=None, candidate=None):
        start_time = time.time()

        prompt, reference, candidate, error_type = await self.make_prompt(
            reference, candidate
        )

        if any(var is None for var in [prompt, reference, candidate]):
            print("Skipping due to None values")
            return

        if len(candidate) < 5 or len(reference) < 5:
            print("Skipping due to short reports")
            return

        try:
            cost, completion = await call_gpt4_turbo("", prompt, temperature=0, n=1)
            end_time = time.time()

            item = {
                "key": {"candidate": candidate, "reference": reference},
                "prompt": prompt,
                "origin": error_type,
                "response": completion,
            }
            self.dataset.append(item)

            with open(OUTPUT_PATH, "w") as file:
                json_string = json.dumps(self.dataset, indent=4)
                file.write(json_string)

            self.cost += cost
            self.elapsed_time += end_time - start_time
            print(f"Generated response. Dataset size: {len(self.dataset)}")
        except Exception as e:
            print(f"Error in get_response: {e}")

    async def generate_dataset(self, number_of_samples=3):
        for i in range(number_of_samples):
            print(f"Generating dataset: {i}/{number_of_samples}")
            await self.get_response()
            if i % 10 == 0:  # Save progress every 10 samples
                print(f"Progress: {i}/{number_of_samples}")
        print("Done!")

    async def ask_GPT4(self):
        print("Starting to ask GPT4...")
        for i, (ref, can) in enumerate(zip(self.reference, self.candidates)):
            await self.get_response(ref, can)
            if i % 10 == 0:  # Print progress every 10 samples
                print(f"Progress: {i}/{len(self.reference)}")
        print("Done!")


if __name__ == "__main__":
    dataset = GREENDataset(INPUT_PATH, column_names=column_names)

    async def main():
        if generate:
            await dataset.generate_dataset(NUM_SAMPLES)
        else:
            await dataset.ask_GPT4()
        print("Dataset generated successfully!")

    asyncio.run(main())
