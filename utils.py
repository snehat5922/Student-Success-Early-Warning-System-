def calculate_risk_score(row):

    attendance = row["Attendance"]
    assignment = row["Assignment_Completion"]
    quiz = row["Quiz_Score"]

    engagement = min(
        row["LMS_Clicks"] / 20,
        100
    )

    score = (
        (100 - attendance) * 0.35 +
        (100 - assignment) * 0.25 +
        (100 - quiz) * 0.20 +
        (100 - engagement) * 0.20
    )

    return round(score, 2)


def recommendation(row):

    if row["Attendance"] < 70:
        return "Attendance Counseling"

    elif row["Quiz_Score"] < 60:
        return "Topic Revision Session"

    elif row["LMS_Clicks"] < 500:
        return "Increase LMS Participation"

    else:
        return "Performing Well"
