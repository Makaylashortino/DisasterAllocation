import pandas as pd

df = pd.read_csv("Data/PublicAssistanceFundedProjectsDetails.csv")

df.columns = df.columns.str.strip()

cols = [
    "stateAbbreviation",
    "declarationDate",
    "incidentType",
    "projectAmount",
    "totalObligated"
]
df = df[cols]


df = df.dropna(subset=["stateAbbreviation"])
df["stateAbbreviation"] = df["stateAbbreviation"].astype(str).str.strip()

df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")
df = df.dropna(subset=["declarationDate"])
df = df[df["declarationDate"] >= "2015-01-01"]
df["declarationDate"] = df["declarationDate"].dt.date

df["projectAmount"] = pd.to_numeric(df["projectAmount"], errors="coerce").fillna(0)
df["totalObligated"] = pd.to_numeric(df["totalObligated"], errors="coerce").fillna(0)

df = df[(df["projectAmount"] >= 0) & (df["totalObligated"] >= 0)]
df = df[(df["projectAmount"] > 0) | (df["totalObligated"] > 0)]

state_totals = (
    df.groupby("stateAbbreviation", as_index=False)
    .agg(
        total_project_amount=("projectAmount", "sum"),
        total_obligated=("totalObligated", "sum"),
        num_projects=("stateAbbreviation", "count")
    )
)

df.to_csv("Data/PublicAssistanceFundedProjectsCleaned.csv", index=False)
state_totals.to_csv("Data/FEMA_State_Totals.csv", index=False)

print(df.head())
print(state_totals.head())