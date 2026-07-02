from connect import *
import math
import re
import datetime


# === Step 1: CT 설정 및 ROI Type 보정 ===
now = datetime.datetime.now()
today = now.strftime('%y%m%d')
examination = get_current("Examination")
patient = get_current("Patient")
case = get_current("Case")
type = "2.5D"

# plan paremeter
plan_name = f"LT_{type}_{today}"

ANGLE_TABLE = {
    "000": {"gantry": {"start": 290, "end": 159}, "colli": {"beam1": 5, "beam2": 354}},
    "011": {"gantry": {"start": 288, "end": 154}, "colli": {"beam1": 5, "beam2": 354}},
    "014": {"gantry": {"start": 287, "end": 155}, "colli": {"beam1": 5, "beam2": 354}},
    "018": {"gantry": {"start": 290, "end": 158}, "colli": {"beam1": 5, "beam2": 354}},
    "020": {"gantry": {"start": 288, "end": 152}, "colli": {"beam1": 5, "beam2": 354}},
    "023": {"gantry": {"start": 290, "end": 155}, "colli": {"beam1": 5, "beam2": 354}},
    "026": {"gantry": {"start": 290, "end": 161}, "colli": {"beam1": 5, "beam2": 354}},
    "038": {"gantry": {"start": 289, "end": 159}, "colli": {"beam1": 5, "beam2": 354}},
    "053": {"gantry": {"start": 290, "end": 156}, "colli": {"beam1": 5, "beam2": 354}},
    "059": {"gantry": {"start": 288, "end": 152}, "colli": {"beam1": 5, "beam2": 354}},
    "060": {"gantry": {"start": 291, "end": 157}, "colli": {"beam1": 5, "beam2": 354}},
    "065": {"gantry": {"start": 292, "end": 160}, "colli": {"beam1": 5, "beam2": 354}},
    "066": {"gantry": {"start": 289, "end": 160}, "colli": {"beam1": 5, "beam2": 354}},
    "069": {"gantry": {"start": 292, "end": 163}, "colli": {"beam1": 5, "beam2": 354}},
    "098": {"gantry": {"start": 286, "end": 151}, "colli": {"beam1": 5, "beam2": 354}},
    "078": {"gantry": {"start": 289, "end": 154}, "colli": {"beam1": 5, "beam2": 354}},
    "082": {"gantry": {"start": 288, "end": 156}, "colli": {"beam1": 5, "beam2": 354}},
    "091": {"gantry": {"start": 290, "end": 160}, "colli": {"beam1": 5, "beam2": 354}},
    "093": {"gantry": {"start": 290, "end": 157}, "colli": {"beam1": 5, "beam2": 354}},
    "096": {"gantry": {"start": 288, "end": 154}, "colli": {"beam1": 5, "beam2": 354}},
}


def extract_wl_number(patient, case): # patient_number 비교를 위한 추출 함수
    for s in [getattr(patient, "PatientID", ""), getattr(patient, "Name", ""),
              getattr(case, "CaseName", ""), getattr(case, "CaseDescription", "")]:
        if not s: 
            continue
        m = re.search(r"(\d+)\s*WL", s, flags=re.IGNORECASE)
        if m: return int(m.group(1))
    return None


with CompositeAction('Apply image set properties'):
    examination.EquipmentInfo.SetImagingSystemReference(ImagingSystemName="Canon_EXLB_2022")

# ROI 이름이 "Couch_Lat"인 경우 삭제
roi_names = [roi.Name for roi in case.PatientModel.RegionsOfInterest]
if "Couch_Lat" in roi_names:
    with CompositeAction('Delete ROI Couch_Lat'):
        case.PatientModel.RegionsOfInterest["Couch_Lat"].DeleteRoi()
        print('"Couch_Lat" ROI가 삭제되었습니다.')
else:
    print('"Couch_Lat" ROI가 존재하지 않습니다.')
    
if "CD_CTV" in roi_names:
    with CompositeAction('Delete ROI CD_CTV'):
        case.PatientModel.RegionsOfInterest["CD_CTV"].DeleteRoi()
        print('"CD_CTV" ROI가 삭제되었습니다.')
else:
    print('"CD_CTV" ROI가 존재하지 않습니다.')
    

# ROI Type이 CTV로 설정되었는지 확인 및 설정
for roi_name in ["CTV_WB"]:
    if roi_name in roi_names:
        roi = case.PatientModel.RegionsOfInterest[roi_name]
        print('roi name : ', roi_name, 'roi type : ', roi.Type)
        if roi.Type != "Ctv":
            with CompositeAction(f'Set ROI {roi_name} Type to CTV'):
                roi.Type = "Ctv"
                print(f'ROI "{roi_name}"의 Type이 CTV로 설정되었습니다.')
        else:
            print(f'ROI "{roi_name}"의 Type은 이미 CTV로 설정되어 있습니다.')
    else:
        print(f'ROI "{roi_name}"이 존재하지 않습니다.')

# Plan 이름 변경
for plan in case.TreatmentPlans:
    if plan.Name == "Empty plan":
        plan.Name = f"LT_PD_{type}_{today}"
        print(f'Plan 이름이 {plan.Name}으로 변경되었습니다.')

# 현재 상태 저장
patient.Save()
print('현재 상태가 저장되었습니다.')

# Plan 생성
# 기존 Plan 삭제
existing_plan = next((p for p in case.TreatmentPlans if p.Name == plan_name), None)
if existing_plan:
    with CompositeAction(f"Delete existing plan '{plan_name}'"):
        print(f"Existing plan '{plan_name}' has been deleted.")

# 새로운 Plan 생성
with CompositeAction('Add treatment plan'):
    retval_0 = case.AddNewPlan(PlanName=plan_name, PlannedBy="", Comment="", ExaminationName=examination.Name, IsMedicalOncologyPlan=False, AllowDuplicateNames=False)
    retval_1 = retval_0.AddNewBeamSet(Name=plan_name, ExaminationName=examination.Name, MachineName="ELT33V", Modality="Photons", TreatmentTechnique="VMAT", PatientPosition="HeadFirstSupine",
                                      NumberOfFractions=5, CreateSetupBeams=True, UseLocalizationPointAsSetupIsocenter=False, UseUserSelectedIsocenterSetupIsocenter=False, Comment="")

print('done - plan created')


# === Step 2: isocenter_CTV 생성 ===

TARGET_ROI_NAME = "CTV_WB"
EXTERNAL_ROI_NAME = "External"
ISO_POI_NAME = "isocenter_CTV"
MAX_DISTANCE_MM = 12.0

structure_set = case.PatientModel.StructureSets[examination.Name]

def has_named_roi_with_contours(ss, roi_name):
    for roi in ss.RoiGeometries:
        if roi.OfRoi.Name == roi_name and roi.HasContours():
            return True
    return False

if not has_named_roi_with_contours(structure_set, TARGET_ROI_NAME):
    raise Exception(f"ROI '{TARGET_ROI_NAME}' not found or has no contours.")
if not has_named_roi_with_contours(structure_set, EXTERNAL_ROI_NAME):
    raise Exception(f"ROI '{EXTERNAL_ROI_NAME}' not found or has no contours.")

def roi_center_x(ss, roi_name):
    box = ss.RoiGeometries[roi_name].GetBoundingBox()
    return (box[0].x + box[1].x) / 2

def roi_center_y(ss, roi_name):
    box = ss.RoiGeometries[roi_name].GetBoundingBox()
    return (box[0].y + box[1].y) / 2

# 중심 계산
target_center = structure_set.RoiGeometries[TARGET_ROI_NAME].GetCenterOfRoi()
external_center_x = roi_center_x(structure_set, EXTERNAL_ROI_NAME)
external_center_y = roi_center_y(structure_set, EXTERNAL_ROI_NAME)

dx = target_center.x - external_center_x
dy = target_center.y - external_center_y
distance = math.sqrt(dx**2 + dy**2)

if distance > MAX_DISTANCE_MM:
    dx = dx * MAX_DISTANCE_MM / distance
    dy = dy * MAX_DISTANCE_MM / distance

iso_x = external_center_x + dx
iso_y = external_center_y + dy
iso_z = target_center.z

# 기존 POI 삭제 후 생성
# existing_pois = [p.OfPoi.Name for p in structure_set.PoiGeometries]
# if ISO_POI_NAME in existing_pois:
#     case.PatientModel.DeletePoi(PoiName=ISO_POI_NAME)
existing_pois = [p.OfPoi.Name for p in structure_set.PoiGeometries]
if ISO_POI_NAME in existing_pois:
    print(f"{ISO_POI_NAME} POI already exists. Deleting it.")
    case.PatientModel.PointsOfInterest['isocenter_CTV'].DeleteRoi()


# POI 생성
poi = case.PatientModel.CreatePoi(Examination=examination, Point={ 'x': iso_x, 'y': iso_y, 'z': iso_z }, Name=ISO_POI_NAME, Color="Red", VisualizationDiameter=1, Type="Isocenter")

# 좌표 수동 설정 (GUI 좌표 보장용)
structure_set.PoiGeometries[ISO_POI_NAME].Point = {'x': iso_x, 'y': iso_y, 'z': iso_z}

print(f"[INFO] POI '{ISO_POI_NAME}' created at x={iso_x:.2f}, y={iso_y:.2f}, z={iso_z:.2f}")

# case = get_current("Case")
# beam_set = get_current("BeamSet") # beam 정보 이건 불확실함

beam_set = next((bs for plan in case.TreatmentPlans 
                 if plan.Name == plan_name 
                 for bs in plan.BeamSets 
                 if bs.DicomPlanLabel == plan_name), None)


## Step 3: Beam 생성 (LT_2D_angle)

patient_number = patient.PatientID
patient_number = patient.PatientID.split('WL')[0]  # "0nn" 형태로 추출
print(f"start -- patient_number: {patient_number}")
wl_number = extract_wl_number(patient, case)
spec = ANGLE_TABLE[patient_number] #"혹시라도 이거 안되면 patient_number로 다시 바꾸기"

start = float(spec["gantry"]["start"])       # 시작 각도
end   = float(spec["gantry"]["end"])         # 끝 각도
col1  = float(spec["colli"]["beam1"])        # arc1 collimator
col2  = float(spec["colli"]["beam2"])        # arc2 collimator

print(f"wl_number: {wl_number}, start: {start}, end: {end}, col1: {col1}, col2: {col2}")

# --- 2) 아크 생성: arc1 = CW(start→end), arc2 = CCW(end→start) ---
# arc1 (CW)
with CompositeAction('Add beam (arc1_LT, beam set: RT_FB_25D_ANG)'):
    b1 = beam_set.CreateArcBeam(
        ArcStopGantryAngle=end,                 # 끝 각도
        ArcRotationDirection="Clockwise",       # 회전 방향
        BeamQualityId="6",
        GimbalPanAngle=0,
        GimbalTiltAngle=0,
        IsocenterData={
            'Position': {'x': iso_x, 'y': iso_y, 'z': iso_z},
            'NameOfIsocenterToRef': "",
            'Name': f"LT_FB_{type}_ang 1",
            'Color': "98, 184, 234"
        },
        Name="arc1_cw",
        Description="",
        GantryAngle=start,                      # 시작 각도
        CouchRotationAngle=0,
        CouchPitchAngle=0,
        CouchRollAngle=0,
        CollimatorAngle=col1                    # 콜리메이터
    )
    b1.SetBolus(BolusName="")

# arc2 (CCW)
with CompositeAction('Add beam (arc2_LT, beam set: RT_FB_25D_ANG)'):
    b2 = beam_set.CreateArcBeam(
        ArcStopGantryAngle=start,               # 끝 각도 = arc1의 시작
        ArcRotationDirection="CounterClockwise",
        BeamQualityId="6",
        GimbalPanAngle=0,
        GimbalTiltAngle=0,
        IsocenterData={
            'Position': {'x': iso_x, 'y': iso_y, 'z': iso_z},
            'NameOfIsocenterToRef': "",
            'Name': f"LT_FB_{type}_ang 2",
            'Color': "98, 184, 234"
        },
        Name="arc2_ccw",
        Description="",
        GantryAngle=end,                        # 시작 각도 = arc1의 끝
        CouchRotationAngle=0,
        CouchPitchAngle=0,
        CouchRollAngle=0,
        CollimatorAngle=col2
    )
    b2.SetBolus(BolusName="")

  
print("done")