import os
import random
import json
from faker import Faker
from babel.dates import format_date
from tqdm import tqdm
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'dd MMM YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY'
           ]

LOCALES = ['vi_VN']
DATA_FOLDERS = os.getcwd()


def create_date():
    """
        Create some fake dates
    """
    date_time = fake.date_object()
    format=random.choice(FORMATS)
    a=0#short
    if format in ['dd MMM YYY','d MMMM, YYY','dd, MMM YYY','d MMMM YYY','long','medium']:
        a=1
    elif format is 'full':
        a=2
    

    try:
        human = format_date(date_time,
                            format=format,
                            locale=random.choice(LOCALES))
        case_change = random.randint(0, 1)
        if case_change == 0:
            human = human.upper()
        elif case_change == 1:
            human = human.lower()
            if a==1:
                human="ngày "+ human
            elif a==2:
                index=human.find(",")
                human=human[:index+1]+" ngày "+human[index+1:]
    


        machine = date_time.isoformat()

    except AttributeError as e:
        # print(e)
        return None, None, None

    return human, machine, date_time


def generate_dataset(dataset_name, n_example):
    with open(dataset_name, 'w') as f:
        for _ in tqdm(range(n_example)):
            human, machine, _ = create_date()
            if human is not None:
                f.write('"' + human + '","' + machine + '"\n')


if __name__ == "__main__":
    data_folder = 'Data'
    path = os.path.join(os.getcwd(), data_folder)
    if not os.path.exists(path):
        os.mkdir(path)

    os.chdir(path)
    print("Starting create dataset:")
    generate_dataset('data2.csv', n_example=1000)
    print("Done!")
    os.chdir('..')
