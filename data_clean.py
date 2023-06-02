import numpy as np
import json
import csv

PERSONAL_PROFILE = [(48, 195, "male"),
                    (60, 180, "male"),
                    (25, 184, "male"),
                    (26, 163, "female"),
                    (35, 176, "male"),
                    (42, 179, "male"),
                    (26, 177, "male"),
                    (27, 186, "male"),
                    (26, 180, "male"),
                    (38, 179, "female"),
                    (25, 171, "female"),
                    (27, 178, "male"),
                    (31, 183, "male"),
                    (45, 181, "male"),
                    (54, 180, "male"),
                    (23, 182, "male")]


def load_data():
    sleep = []
    score = []
    active = []
    for i in range(1, 17):
        if i == 12:
            continue

        with open(f"./data/pmdata/p{i:02}/fitbit/sleep.json") as f:
            json_sleep = json.load(f)
            temp_sleep = []
            for j in range(len(json_sleep)):
                temp_sleep.append(
                    [json_sleep[j]["dateOfSleep"], json_sleep[j]["startTime"][11:19], json_sleep[j]["endTime"][11:19],
                     json_sleep[j]["duration"]])
        with open(f"./data/pmdata/p{i:02}/fitbit/sleep_score.csv") as f:
            csv_reader = csv.reader(f, delimiter=',')
            tag_read = False
            temp_score = []
            for row in csv_reader:
                if not tag_read:
                    tag_read = True
                    continue
                temp_score.append([row[0][:10], int(row[1]), int(row[2])])
        with open(f"./data/pmdata/p{i:02}/fitbit/lightly_active_minutes.json") as f:
            json_active_l = json.load(f)
            active_l = []
            for j in range(len(json_active_l)):
                active_l.append([json_active_l[j]["dateTime"][:10], int(json_active_l[j]["value"])])
        with open(f"./data/pmdata/p{i:02}/fitbit/moderately_active_minutes.json") as f:
            json_active_m = json.load(f)
            active_m = []
            for j in range(len(json_active_m)):
                active_m.append([json_active_m[j]["dateTime"][:10], int(json_active_m[j]["value"])])
        with open(f"./data/pmdata/p{i:02}/fitbit/very_active_minutes.json") as f:
            json_active_v = json.load(f)
            active_v = []
            for j in range(len(json_active_v)):
                active_v.append([json_active_v[j]["dateTime"][:10], int(json_active_v[j]["value"])])

        # sleep match
        last_sleep_id = -1
        score_i = []
        for j in range(len(temp_sleep)):
            found = False
            for k in range(last_sleep_id + 1, len(temp_score)):
                if temp_sleep[j][0] == temp_score[k][0]:
                    score_i.append(temp_score[k][2])
                    found = True
                    last_sleep_id = k
                    break
            if not found:
                score_i.append(-1)

        # activity match
        last_active_id = -1
        active_i = []
        for j in range(len(temp_sleep)):
            found = False
            for k in range(last_active_id + 1, len(active_l)):
                if temp_sleep[j][0] == active_l[k][0]:
                    active_i.append((active_l[k][1], active_m[k][1], active_v[k][1]))
                    found = True
                    last_active_id = k
                    break
            if not found:
                active_i.append((-1, -1, -1))

        # filter
        temp_sleep, score_i, active_i = filter_data(temp_sleep, score_i, active_i)

        # append to all
        sleep.append(temp_sleep)
        score.append(score_i)
        active.append(active_i)

    data = []
    for i in range(len(sleep)):
        temp_data = []
        for j in range(1, len(sleep[i])):
            temp_time = [int(k) for k in sleep[i][j - 1][1].split(':')]
            temp_time = temp_time[0] * 60 + temp_time[1] + temp_time[2] / 60
            temp_time2 = [int(k) for k in sleep[i][j][0].split(':')]
            temp_time2 = temp_time2[0] * 60 + temp_time2[1] + temp_time2[2] / 60
            if temp_time2 > 720:
                temp_time2 = temp_time2-1440
            temp_store = [temp_time, temp_time2, sleep[i][j][2] / 60000,
                          active[i][j][0] + active[i][j][1] + active[i][j][2]]
            temp_data.append(temp_store)
        data.append(temp_data)

    return data


def filter_data(sleep, score, active):
    i = 0
    while i < len(sleep):
        if score[i] < 70:
            del sleep[i]
            del score[i]
            del active[i]
        else:
            sleep[i] = sleep[i][1:]
            i += 1
    return sleep, score, active


def load_personal():
    data = []
    for i in range(16):
        if i == 11:
            continue
        data.append([PERSONAL_PROFILE[i][0] / 100
                    , PERSONAL_PROFILE[i][1] / 300
                    , -0.5 if PERSONAL_PROFILE[i][2] == "male" else 0.5])
    return data


def merge_data(data, personal):
    merge = []
    output = []
    for i in range(15):
        for j in range(len(data[i])):
            inner_merge = []
            for k in range(len(personal[i])):
                inner_merge.append(personal[i][k])
            for k in range(len(data[i][j])):
                if not k == 1:
                    inner_merge.append(data[i][j][k])
                else:
                    output.append([data[i][j][k]])
            merge.append(inner_merge)

    for i in range(len(merge[0])):
        merge[:][i] /= np.linalg.norm(merge[:][i])
    output /= np.linalg.norm(output)

    return merge, output


def clean_data():
    data = load_data()  # prev_sleep_end(min), sleep_start(min from 12AM), sleep_duration(min), activity(min)
    personal = load_personal()  # age, weight, gender (male-0)
    merge, output = merge_data(data, personal)
    return merge, output


def save_npy(data, output):
    np.save('./data/pmdata_clean/data.npy', data)
    np.save('./data/pmdata_clean/output.npy', output)
    # for i in range(1, 12):
    #     temp_data = np.asarray(data[i-1], dtype='f')
    #     temp_output = np.asarray(output[i-1], dtype='f')
    #     np.save(f'./data/pmdata_clean/data{i:02}.npy', temp_data)
    #     np.save(f'./data/pmdata_clean/output{i:02}.npy', temp_output)
    # for i in range(13, 17):
    #     temp_data = np.asarray(data[i-2], dtype='f')
    #     temp_output = np.asarray(output[i-2], dtype='f')
    #     np.save(f'./data/pmdata_clean/data{i:02}.npy', temp_data)
    #     np.save(f'./data/pmdata_clean/output{i:02}.npy', temp_output)


def main():
    data, output = clean_data()
    save_npy(data, output)


if __name__ == "__main__":
    main()
