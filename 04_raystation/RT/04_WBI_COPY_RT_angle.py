# -*- coding: utf-8 -*-
import logging
import time
import math
import threading
import os
import hashlib
import json
import datetime
import re

from connect import *

from tkinter import *
from tkinter import messagebox

import numpy as np  # numpy 사용

now = datetime.datetime.now()
today = now.strftime('%y%m%d')

RAYSTATION_ANONYMIZE = os.environ.get("WBI_RAYSTATION_ANONYMIZE", "true").lower() in ("1", "true", "yes")
LOG_PATIENT_IDENTIFIERS = os.environ.get("WBI_LOG_PATIENT_IDENTIFIERS", "false").lower() in ("1", "true", "yes")

def safe_case_label(patient):
    source = str(getattr(patient, "PatientID", "unknown"))
    return "case_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
ab_path = os.path.join(os.environ.get("WBI_EXPORT_ROOT", r"C:\WBI_exports"), today, "RT")

# Export할 폴더(공유 경로) 지정

# === 4. export dose dicom =====
def LogWarning(error):
    """Non-blocking warning 발생 시 메시지를 로그로 출력"""
    try:
        jsonWarnings = json.loads(str(error))
        print("WARNING! Export Aborted!")
        print("Comment:")
        print(jsonWarnings["Comment"])
        print("Warnings:")
        for w in jsonWarnings["Warnings"]:
            print(w)
    except ValueError:
        print("Error occurred. Could not export.")

def LogCompleted(result):
    """Export 성공 시 메시지를 로그로 출력"""
    try:
        jsonWarnings = json.loads(str(result))
        print("Completed!")
        print("Comment:")
        print(jsonWarnings["Comment"])
        print("Warnings:")
        for w in jsonWarnings["Warnings"]:
            print(w)
        print("Export notifications:")
        for w in jsonWarnings["Notifications"]:
            print(w)
    except ValueError:
        print("Error reading completion messages.")

def export_dicom():
    # =====================================================================
    # 메인 스크립트 시작
    # =====================================================================
    # ClinicDB, Case, Examination 등 RayStation 객체 가져오기
    clinic_db   = get_current("ClinicDB")
    case        = get_current("Case")
    examination = get_current("Examination")
    beam_set    = get_current("BeamSet")
    patient     = get_current("Patient")
    # =====================================================================
    patient_folder_name = safe_case_label(patient)
    print(patient_folder_name) # 0nn patient_number
    model_name = case.TreatmentPlans[3].Name + "_M_RTDOSE" # 3번째 항목 이름  LT_FB_{model_name}_OP
    path = os.path.join(ab_path, patient_folder_name, model_name)

    # 폴더가 없으면 생성
    if not os.path.exists(path):
        os.makedirs(path)

    # RayStation에서 환자 데이터 저장
    patient.Save()

    # Default AnonymizationSettings 불러와서, 익명화 기본값(Anonymize=True) 설정
    default_anonymization_options = clinic_db.GetSiteSettings().DicomSettings.DefaultAnonymizationOptions
    anonymization_settings = {
        "Anonymize": RAYSTATION_ANONYMIZE,
        "AnonymizedName": patient_folder_name,
        "AnonymizedID": patient_folder_name,
        "RetainDates": default_anonymization_options.RetainLongitudinalTemporalInformationFullDatesOption,
        "RetainDeviceIdentity": default_anonymization_options.RetainDeviceIdentityOption,
        "RetainInstitutionIdentity": default_anonymization_options.RetainInstitutionIdentityOption,
        "RetainUIDs": default_anonymization_options.RetainUIDs,
        "RetainSafePrivateAttributes": default_anonymization_options.RetainSafePrivateOption
    }

    try:
        # 1차 Export 시도 (IgnorePreConditionWarnings=False)
        result = case.ScriptableDicomExport(
            ExportFolderPath=path,
            AnonymizationSettings=anonymization_settings,
            PhysicalBeamSetDoseForBeamSets=[beam_set.BeamSetIdentifier()],
            DicomFilter="",
            IgnorePreConditionWarnings=False
        )
        LogCompleted(result)

    except System.InvalidOperationException as error:
        # Non-blocking Warning 또는 Blocking Warning으로 인한 실패
        LogWarning(error)

        print("\nTrying to export again with IgnorePreConditionWarnings=True\n")

        # 2차 Export 시도 (IgnorePreConditionWarnings=True)
        result = case.ScriptableDicomExport(
            ExportFolderPath=path,
            AnonymizationSettings=anonymization_settings,
            PhysicalBeamSetDoseForBeamSets=[beam_set.BeamSetIdentifier()],
            DicomFilter="",
            IgnorePreConditionWarnings=True
        )
        LogCompleted(result)

    except Exception as e:
        print("Except %s" % e)

# copy plan
def copy_plan():
    case = get_current("Case")
    model_name = case.TreatmentPlans[3].Name
    print(model_name)

    print("export_mimicking_dicom()")
    export_dicom()

    case.CopyPlan(PlanName=model_name, NewPlanName=model_name+"_OP", KeepBeamSetNames=False)

print("copy plan start!")
copy_plan()
