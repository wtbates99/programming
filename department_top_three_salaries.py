import pandas as pd

employee_df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7],
    "name": ["Joe", "Henry", "Sam", "Max", "Janet", "Randy", "Will"],
    "salary": [85000, 80000, 60000, 90000, 69000, 85000, 70000],
    "departmentId": [1, 2, 2, 1, 1, 1, 1]
})

department_df = pd.DataFrame({
    "id": [1, 2],
    "name": ["IT", "Sales"]
})


def top_three_salaries(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    employee = employee.sort_values(by=["salary"], ascending=False)
    deparment = department.rename(columns={'id': 'departmentId', 'name': 'department'})
    employee = employee.merge(deparment, on="departmentId")
    frames = []

id   name  salary  departmentId department
    for dept in department["id"]:
        e = employee[employee['departmentId'] == dept]

        salaries = e["salary"].unique()
        if len(salaries) <= 3:
            lowest = salaries[-1]
        else:
            lowest = salaries[2]

        filtered_df = e[e['salary'] >= lowest]
        frames.append(filtered_df)

    frames = pd.concat(frames)
    return frames


top_three_salaries(employee_df, department_df)
