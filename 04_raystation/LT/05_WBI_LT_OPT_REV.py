# -*- coding: utf-8 -*-

import logging
import time
import math
import threading
import os
import hashlib
import System
import json
import traceback
import datetime

from connect import *
from connect import CompositeAction

from tkinter import *
from tkinter import messagebox

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 현재 세션의 객체 가져오기
now = datetime.datetime.now()
today = now.strftime('%y%m%d')

RAYSTATION_ANONYMIZE = os.environ.get("WBI_RAYSTATION_ANONYMIZE", "true").lower() in ("1", "true", "yes")
LOG_PATIENT_IDENTIFIERS = os.environ.get("WBI_LOG_PATIENT_IDENTIFIERS", "false").lower() in ("1", "true", "yes")
CLINICAL_GOAL_TEMPLATE = os.environ.get("WBI_CLINICAL_GOAL_TEMPLATE", "YOUR_CLINICAL_GOAL_TEMPLATE")

def safe_case_label(patient):
    source = str(getattr(patient, "PatientID", "unknown"))
    return "case_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
case = get_current("Case")
examination = get_current("Examination")
patient = get_current("Patient")
patient_db = get_current("PatientDB")
ui = get_current("ui")

SCRIPT_START_TIME = None
selected_goals_global = None

ab_path = os.path.join(os.environ.get("WBI_EXPORT_ROOT", r"C:\WBI_exports"), today, "LT")

# === DICOM Export =====
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

# Export할 폴더(공유 경로) 지정
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
    model_DOSE_name = case.TreatmentPlans[4].Name + "_RTDOSE" # 3번째 항목 이름  LT_FB_{model_name}_OP_RTDOSE
    model_PLAN_name = case.TreatmentPlans[4].Name + "_RTPLAN" # 3번째 항목 이름  LT_FB_{model_name}_OP_RTPLAN
    path = os.path.join(ab_path, patient_folder_name, model_DOSE_name)
    path_plan =  os.path.join(ab_path, patient_folder_name, model_PLAN_name)

    # 폴더가 없으면 생성
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(path_plan):
        os.makedirs(path_plan)

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

    print("OPT DOSE DICOM EXPORT")

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

    print("OPT PLAN DICOM EXPORT")

    try:
        # 1차 Export 시도 (IgnorePreConditionWarnings=False)
        result = case.ScriptableDicomExport(
            ExportFolderPath=path_plan,
            AnonymizationSettings=anonymization_settings,
            BeamSets = [beam_set.BeamSetIdentifier()],
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
            ExportFolderPath=path_plan,
            AnonymizationSettings=anonymization_settings,
            BeamSets = [beam_set.BeamSetIdentifier()],
            DicomFilter="",
            IgnorePreConditionWarnings=True
        )
        LogCompleted(result)

    except Exception as e:
        print("Except %s" % e)

#====================================================================
def center_window(window, window_width=None, window_height=None):
    """창을 화면 중앙으로 이동시키는 함수"""
    window.update_idletasks()
    if window_width is None:
        window_width = window.winfo_width()
    if window_height is None:
        window_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")


def debug_goal_attributes_complete(plan):
    """Clinical Goal 객체의 모든 속성을 완전 분석"""
    try:
        goals = get_clinical_goals_safely(plan)
        if not goals:
            print("Goals를 찾을 수 없습니다.")
            return

        goal = goals[0]
        print("="*50)
        print("GOAL 객체 완전 분석")
        print("="*50)

        print(f"Goal 타입: {type(goal)}")
        print(f"ROI 이름: {goal.ForRegionOfInterest.Name}")

        print("\n--- Goal 객체의 모든 속성 ---")
        goal_attrs = [attr for attr in dir(goal) if not attr.startswith('_')]
        for attr in goal_attrs:
            try:
                value = getattr(goal, attr)
                if not callable(value):
                    print(f"goal.{attr}: {value} (타입: {type(value)})")
            except:
                print(f"goal.{attr}: 접근 불가")

        if hasattr(goal, "PlanningGoal"):
            print("\n--- PlanningGoal 객체의 모든 속성 ---")
            pg = goal.PlanningGoal
            print(f"PlanningGoal 타입: {type(pg)}")

            pg_attrs = [attr for attr in dir(pg) if not attr.startswith('_')]
            for attr in pg_attrs:
                try:
                    value = getattr(pg, attr)
                    if not callable(value):
                        print(f"PlanningGoal.{attr}: {value} (타입: {type(value)})")
                except:
                    print(f"PlanningGoal.{attr}: 접근 불가")

        print("\n--- 현재값 확인 ---")
        try:
            current_val = goal.GetClinicalGoalValue()
            print(f"GetClinicalGoalValue(): {current_val}")
        except Exception as e:
            print(f"GetClinicalGoalValue() 오류: {e}")

    except Exception as e:
        print(f"디버깅 중 오류: {str(e)}")


def debug_all_goals_brief(plan):
    """모든 Clinical Goals의 간략한 정보 출력"""
    try:
        goals = get_clinical_goals_safely(plan)
        if not goals:
            print("Goals를 찾을 수 없습니다.")
            return

        print("\n" + "="*80)
        print(f"총 {len(goals)}개의 Clinical Goals 발견")
        print("="*80)

        for i, goal in enumerate(goals):
            try:
                roi_name = "Unknown"
                if hasattr(goal, "ForRegionOfInterest") and goal.ForRegionOfInterest:
                    roi_name = goal.ForRegionOfInterest.Name

                planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                goal_type = getattr(planning_goal, "Type", "Unknown")
                goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)
                parameter_value = getattr(planning_goal, "ParameterValue", None)

                current_value = None
                try:
                    current_value = goal.GetClinicalGoalValue()
                except:
                    pass

                is_met = None
                if current_value is not None and acceptance_level > 0:
                    try:
                        if goal_criteria == "AtMost":
                            is_met = current_value <= acceptance_level
                        elif goal_criteria == "AtLeast":
                            is_met = current_value >= acceptance_level
                    except:
                        pass

                status_str = ""
                if is_met is not None:
                    status_str = "✓ 달성" if is_met else "✗ 미달성"

                print(f"\nGoal {i+1}: {roi_name}")
                print(f"  Type: {goal_type}")
                print(f"  Criteria: {goal_criteria}")
                print(f"  Acceptance: {acceptance_level:.2f}")
                if parameter_value is not None:
                    print(f"  Parameter: {parameter_value:.2f}")
                if current_value is not None:
                    print(f"  Current: {current_value:.2f}")
                if status_str:
                    print(f"  Status: {status_str}")

            except Exception as e:
                print(f"\nGoal {i+1}: 오류 - {str(e)}")

        print("="*80 + "\n")

    except Exception as e:
        print(f"디버깅 중 오류: {str(e)}")


def debug_goals_status(plan, selected_goals):
    """선택된 목표들의 달성 상태를 상세히 출력"""
    try:
        print("\n" + "="*80)
        print("목표 달성 상태 디버깅")
        print("="*80)

        met_count = 0
        total_count = len(selected_goals)

        for i, goal in enumerate(selected_goals):
            try:
                roi_name = "Unknown"
                if hasattr(goal, "ForRegionOfInterest") and goal.ForRegionOfInterest:
                    roi_name = goal.ForRegionOfInterest.Name

                planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)

                current_value = None
                try:
                    current_value = goal.GetClinicalGoalValue()
                except:
                    pass

                is_met = False
                if current_value is not None and acceptance_level > 0:
                    try:
                        if goal_criteria == "AtMost":
                            is_met = current_value <= acceptance_level
                        elif goal_criteria == "AtLeast":
                            is_met = current_value >= acceptance_level
                    except:
                        pass

                if is_met:
                    met_count += 1

                status_symbol = "✓" if is_met else "✗"
                print(f"{status_symbol} Goal {i+1}: {roi_name} - Current: {current_value:.2f} vs Target: {acceptance_level:.2f} ({goal_criteria})")

            except Exception as e:
                print(f"✗ Goal {i+1}: 오류 - {str(e)}")

        print(f"\n총 달성: {met_count}/{total_count}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"디버깅 중 오류: {str(e)}")


class OptimizationStatusGUI:
    def __init__(self, patient_name, patient_id, plan_name, plan, selected_goals):
        self.root = Tk()
        self.root.title("Auto Planning Status")
        self.root.update_idletasks()
        self.root.attributes("-topmost", True)
        self.start_time = SCRIPT_START_TIME
        self.is_running = True
        self.plan = plan

        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        time_frame = Frame(main_frame)
        time_frame.pack(fill=X, pady=(0, 5))
        self.time_label = Label(
            time_frame, text="소요시간: 00:00:00", font=("Arial", 9, "bold")
        )
        self.time_label.pack(anchor=W)
        self.update_time()

        info_frame = LabelFrame(
            main_frame, text="환자 정보", font=("Arial", 9, "bold"), pady=5, padx=5
        )
        info_frame.pack(fill=X, pady=(0, 10))

        Label(info_frame, text=f"환자명: {patient_name}", font=("Arial", 9)).pack(anchor=W)
        Label(info_frame, text=f"ID: {patient_id}", font=("Arial", 9)).pack(anchor=W)
        Label(info_frame, text=f"Plan: {plan_name}", font=("Arial", 9)).pack(anchor=W)

        status_frame = LabelFrame(
            main_frame, text="최적화 상태", font=("Arial", 9, "bold"), pady=5, padx=5
        )
        status_frame.pack(fill=X, pady=(0, 10))

        self.operation_label = Label(
            status_frame, text="현재 작업: 초기화 중...", font=("Arial", 9), fg="black"
        )
        self.iteration_label = Label(
            status_frame, text="Cycle: 0/100", font=("Arial", 9), fg="black"
        )
        self.goals_label = Label(
            status_frame, text="목표 달성: 0/0", font=("Arial", 9), fg="black"
        )
        self.dose_label = Label(
            status_frame, text="현재 최대 선량: 0.0 cGy", font=("Arial", 9), fg="black"
        )

        self.operation_label.pack(fill=X, pady=1)
        self.iteration_label.pack(fill=X, pady=1)
        self.goals_label.pack(fill=X, pady=1)
        self.dose_label.pack(fill=X, pady=1)

        if selected_goals:
            goals_frame = LabelFrame(
                main_frame,
                text="선택된 목표",
                font=("Arial", 9, "bold"),
                pady=5,
                padx=5,
            )
            goals_frame.pack(fill=BOTH, expand=True)

            canvas = Canvas(goals_frame)
            scrollbar = Scrollbar(goals_frame, orient="vertical", command=canvas.yview)
            self.goals_container = Frame(canvas)

            canvas.configure(yscrollcommand=scrollbar.set)

            scrollbar.pack(side=RIGHT, fill=Y)
            canvas.pack(side=LEFT, fill=BOTH, expand=True)

            canvas.create_window((0, 0), window=self.goals_container, anchor="nw")

            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas_width = max(self.goals_container.winfo_reqwidth(), 600)
                canvas.configure(width=canvas_width)

            self.goals_container.bind("<Configure>", on_frame_configure)

            self.goal_labels = []
            self.goal_list = []

            self.add_goals(selected_goals)

        bottom_container = Frame(main_frame)
        bottom_container.pack(fill=X, pady=(5, 0))

        self.completion_frame = LabelFrame(
            bottom_container,
            text="최적화 결과",
            font=("Arial", 9, "bold"),
            pady=5,
            padx=5,
        )

        self.auto_resize_and_center()
        self.last_update = time.time()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_time(self):
        """소요시간 실시간 업데이트"""
        if not self.is_running:
            return

        try:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.time_label.config(
                text=f"소요시간: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )

            if self.is_running:
                self.root.after(1000, self.update_time)

        except Exception as e:
            logging.error(f"시간 업데이트 중 오류: {str(e)}")
            if self.is_running:
                self.root.after(1000, self.update_time)

    def format_goal_text(self, goal):
        """RayStation 2024A Clinical Goal 객체 정보 읽기"""
        try:
            roi_name = "Unknown"
            try:
                if hasattr(goal, "ForRegionOfInterest") and goal.ForRegionOfInterest:
                    roi_name = goal.ForRegionOfInterest.Name
            except:
                roi_name = "Unknown"

            planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal

            goal_type = getattr(planning_goal, "Type", "Unknown")
            goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
            parameter_value = getattr(planning_goal, "ParameterValue", None)
            acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)

            current_value = None
            try:
                current_value = goal.GetClinicalGoalValue()
            except Exception as e:
                logging.debug(f"GetClinicalGoalValue 오류: {str(e)}")

            is_met = None
            if current_value is not None and acceptance_level > 0:
                try:
                    if goal_criteria == "AtMost":
                        is_met = current_value <= acceptance_level
                    elif goal_criteria == "AtLeast":
                        is_met = current_value >= acceptance_level
                    else:
                        is_met = current_value <= acceptance_level
                except:
                    is_met = None

            text = f"{roi_name} - {goal_type} - {goal_criteria} {acceptance_level:.1f}"

            if goal_type == "DoseAtVolume":
                text += f" cGy (Vol: {parameter_value:.2f}%)" if parameter_value is not None else " cGy"
            elif goal_type == "VolumeAtDose":
                text += f"% (Dose: {parameter_value:.2f} cGy)" if parameter_value is not None else "%"
            elif goal_type == "DoseAtAbsoluteVolume":
                text += f" cGy (Vol: {parameter_value:.2f} cc)" if parameter_value is not None else " cGy"
            elif goal_type == "AverageDose":
                text += " cGy"
            elif goal_type == "ConformityIndex":
                text += f" (RefDose: {parameter_value:.2f} cGy)" if parameter_value is not None else ""
            elif goal_type == "HomogeneityIndex":
                text += f" (RefDose: {parameter_value:.2f} cGy)" if parameter_value is not None else ""
            else:
                text += " cGy"

            if current_value is not None:
                if goal_type == "VolumeAtDose":
                    text += f" | Current: {current_value:.2f}%"
                else:
                    text += f" | Current: {current_value:.1f} cGy"

            return text, is_met

        except Exception as e:
            logging.error(f"format_goal_text 오류: {str(e)}")
            return "오류: 목표 정보를 읽을 수 없습니다.", None

    def add_goals(self, selected_goals):
        """선택된 목표들을 GUI에 추가"""
        self.goal_list = selected_goals

        for i, goal in enumerate(selected_goals):
            try:
                goal_text, is_met = self.format_goal_text(goal)

                if is_met is None:
                    bg_color = "white"
                elif is_met:
                    bg_color = "#90EE90"
                else:
                    bg_color = "#FFB6C1"

                label = Label(
                    self.goals_container,
                    text=goal_text,
                    font=("Arial", 8),
                    anchor=W,
                    bg=bg_color,
                    relief=SOLID,
                    borderwidth=1,
                    padx=5,
                    pady=2,
                )
                label.pack(fill=X, pady=1)
                self.goal_labels.append(label)
            except Exception as e:
                logging.error(f"목표 추가 중 오류 (Goal {i+1}): {str(e)}")
                error_label = Label(
                    self.goals_container,
                    text=f"Goal {i+1}: 오류",
                    font=("Arial", 8),
                    anchor=W,
                    bg="yellow",
                    relief=SOLID,
                    borderwidth=1,
                    padx=5,
                    pady=2,
                )
                error_label.pack(fill=X, pady=1)
                self.goal_labels.append(error_label)

    def update_goals(self):
        """목표 상태 업데이트"""
        try:
            for i, (label, goal) in enumerate(zip(self.goal_labels, self.goal_list)):
                try:
                    goal_text, is_met = self.format_goal_text(goal)

                    if is_met is None:
                        bg_color = "white"
                    elif is_met:
                        bg_color = "#90EE90"
                    else:
                        bg_color = "#FFB6C1"

                    label.config(text=goal_text, bg=bg_color)
                except Exception as e:
                    logging.debug(f"목표 업데이트 중 오류 (Goal {i+1}): {str(e)}")
        except Exception as e:
            logging.error(f"update_goals 오류: {str(e)}")

    def update_status(
        self,
        current_iteration,
        max_iteration,
        goals_met,
        total_goals,
        current_max_dose,
        operation_text="최적화 진행 중",
    ):
        """상태 정보 업데이트"""
        try:
            self.operation_label.config(text=f"현재 작업: {operation_text}")
            self.iteration_label.config(
                text=f"Cycle: {current_iteration}/{max_iteration}"
            )
            self.goals_label.config(text=f"목표 달성: {goals_met}/{total_goals}")
            self.dose_label.config(text=f"현재 최대 선량: {current_max_dose:.1f} cGy")

            self.update_goals()

            self.root.update()
        except Exception as e:
            logging.error(f"update_status 오류: {str(e)}")

    def show_completion_message(
        self, message, final_time="00:00:00", final_d95=0, final_max=0
    ):
        """완료 메시지 표시"""
        try:
            self.is_running = False

            self.completion_frame.pack(fill=X, pady=(10, 0))

            completion_label = Label(
                self.completion_frame,
                text=message,
                font=("Arial", 9),
                justify=LEFT,
                fg="blue",
            )
            completion_label.pack(anchor=W)

            self.time_label.config(text=f"총 소요시간: {final_time}")

            self.auto_resize_and_center()

            self.root.update()
        except Exception as e:
            logging.error(f"show_completion_message 오류: {str(e)}")

    def auto_resize_and_center(self):
        """창 크기를 내용에 맞게 자동 조정하고 중앙에 배치"""
        try:
            self.root.update_idletasks()

            req_width = self.root.winfo_reqwidth()
            req_height = self.root.winfo_reqheight()

            width = max(600, min(req_width, 800))
            height = max(400, min(req_height, 900))

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            self.root.geometry(f"{width}x{height}+{x}+{y}")
        except Exception as e:
            logging.error(f"auto_resize_and_center 오류: {str(e)}")

    def on_closing(self):
        """창 닫기 이벤트 처리"""
        if messagebox.askokcancel("종료", "최적화를 중단하고 창을 닫으시겠습니까?"):
            self.is_running = False
            self.root.destroy()


def get_current_plan():
    """현재 활성화된 Plan 가져오기"""
    try:
        plan = get_current("Plan")
        if plan:
            logging.info(f"현재 활성 Plan: {plan.Name}")
            return plan
        else:
            messagebox.showerror("오류", "현재 활성화된 Plan을 찾을 수 없습니다.")
            return None
    except Exception as e:
        logging.error(f"Plan 가져오기 실패: {str(e)}")
        messagebox.showerror("오류", f"Plan을 가져올 수 없습니다: {str(e)}")
        return None


def setup_prescription(plan):
    """
    Prescription 자동 설정
    CTV_WB을 기준으로 95% volume에 2600cGy (5fx) 설정
    2470cGy 이상이면 만족
    """
    try:
        # Use global patient variable
        global patient

        beam_set = plan.BeamSets[0]

        # 기존 Prescription 확인
        if beam_set.Prescription.PrimaryPrescriptionDoseReference:
            existing_roi = beam_set.Prescription.PrimaryPrescriptionDoseReference.OnStructure.Name
            existing_dose = beam_set.Prescription.PrimaryPrescriptionDoseReference.DoseValue
            logging.info(f"기존 Prescription 존재: {existing_roi}, {existing_dose}cGy")

            # CTV_WB에 2600cGy가 이미 설정되어 있으면 OK
            if existing_roi == "CTV_WB" and existing_dose >= 2470:
                logging.info(f"✅ Prescription 확인 완료: {existing_roi} 95% = {existing_dose}cGy")
                return True
            else:
                logging.warning(f"⚠️ Prescription이 예상과 다릅니다: {existing_roi} {existing_dose}cGy")
                logging.warning(f"⚠️ 권장 설정: CTV_WB 95% = 2600cGy (5fx)")

        # Prescription 자동 설정
        logging.info("=== Prescription 자동 설정 시작 ===")

        # NumberOfFractions는 SetDefaultDoseGrid나 EditPrescription으로 설정
        # 직접 mutation은 불가능하므로 생략하고 prescription만 설정

        # CTV_WB에 대한 Prescription 추가
        with CompositeAction('Add Prescription for CTV_WB'):
            # 새 prescription 추가 (기존 것이 있어도 AddRoiPrescriptionDoseReference가 처리)
            beam_set.AddRoiPrescriptionDoseReference(
                RoiName="CTV_WB",
                DoseVolume=95,  # 95%
                PrescriptionType="DoseAtVolume",
                DoseValue=2600,  # 2600cGy total dose
                RelativePrescriptionLevel=1.0
            )

            logging.info("✅ Prescription 설정 완료: CTV_WB 95% = 2600cGy")
            logging.info("   → 2470cGy 이상이면 Goal 만족")
            logging.info("   ⚠️ NumberOfFractions은 GUI에서 확인 필요 (자동 설정 불가)")

        return True

    except Exception as e:
        logging.error(f"❌ Prescription 자동 설정 실패: {str(e)}")
        logging.error(f"상세 오류: {traceback.format_exc()}")

        # 실패 시 사용자에게 수동 설정 안내
        messagebox.showwarning(
            "Prescription 자동 설정 실패",
            f"Prescription 자동 설정에 실패했습니다.\n\n"
            f"오류: {str(e)}\n\n"
            f"RayStation GUI에서 수동으로 설정해주세요:\n"
            f"1. Plan → Prescription\n"
            f"2. ROI: CTV_WB\n"
            f"3. Type: DoseAtVolume\n"
            f"4. Volume: 95%\n"
            f"5. Dose: 2600 cGy\n"
            f"6. Fractions: 5\n\n"
            f"'OK'를 클릭하면 스크립트를 계속 진행합니다."
        )

        return True  # 수동 설정 후 계속 진행


def load_clinical_goals_from_template(plan, template_name=CLINICAL_GOAL_TEMPLATE):
    """
    Clinical Goal template을 안전하게 불러와서 적용

    Args:
        plan: 현재 plan 객체
        template_name: 불러올 template 이름 (기본값: 'b_dm_g')

    Returns:
        bool: 성공 여부
    """
    # Use global variables
    global patient_db, case

    try:
        logging.info(f"=== Clinical Goal Template 불러오기: '{template_name}' ===")

        # 기존 Clinical Goals는 유지 (삭제하지 않음)
        existing_goals = list(plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions)
        logging.info(f"기존 Clinical Goals: {len(existing_goals)}개")

        # Template 불러오기 - 올바른 API 사용
        try:
            # 사용 가능한 template 목록 조회
            available_templates_info = patient_db.GetClinicalGoalTemplateInfo()
            available_template_names = [t['Name'] for t in available_templates_info if 'Name' in t]

            logging.info(f"사용 가능한 Clinical Goals templates: {available_template_names}")

            # Template이 존재하는지 확인
            if template_name not in available_template_names:
                logging.error(f"❌ Template '{template_name}'을 찾을 수 없습니다.")
                logging.error(f"사용 가능한 templates: {available_template_names}")

                # 사용자에게 수동 설정 안내
                result = messagebox.askyesno(
                    "Clinical Goals Template 필요",
                    f"Template '{template_name}'을 찾을 수 없습니다.\n\n"
                    f"사용 가능한 templates:\n{', '.join(available_template_names)}\n\n"
                    "수동으로 Clinical Goals를 설정하시겠습니까?\n\n"
                    "설정 방법:\n"
                    "1. Plan → Clinical Goals\n"
                    "2. Load Template → 'b_dm_g' 선택\n\n"
                    "'Yes'를 선택하면 스크립트를 계속 진행합니다.\n"
                    "'No'를 선택하면 스크립트를 종료합니다."
                )

                if not result:
                    logging.info("사용자가 스크립트를 취소했습니다.")
                    return False

                logging.info("✅ Clinical Goals 확인 완료 (사용자 설정)")
                return True

            # Template을 직접 Plan에 적용
            logging.info(f"Applying clinical goals template: {template_name}")

            # 먼저 현재 case의 ROI 목록 확인
            case_rois = []
            try:
                for roi in case.PatientModel.RegionsOfInterest:
                    case_rois.append(roi.Name)
                logging.info(f"현재 Case의 ROI 목록 ({len(case_rois)}개): {case_rois}")
            except Exception as e:
                logging.warning(f"ROI 목록 가져오기 실패: {str(e)}")

            # Template 로드
            template = patient_db.LoadTemplateClinicalGoals(
                templateName=template_name,
                lockMode='Read'
            )

            # Screenshot에서 확인한 'b_dm_g' template의 ROI들
            # Template이 ForRegionOfInterest=None으로 저장되어 있으므로
            # 적용 시점에 ROI 매핑을 제공해야 함
            expected_template_rois = ['CTV_WB', 'Contra_Lung', 'Contra_Breast', 'Ipsi_Lung', 'Heart']

            # 현재 Case에 있는 ROI만 매핑
            roi_mapping = {}
            for roi_name in expected_template_rois:
                if roi_name in case_rois:
                    roi_mapping[roi_name] = roi_name
                    logging.info(f"  ✓ ROI 매핑: {roi_name}")
                else:
                    logging.warning(f"  ⚠️ ROI '{roi_name}'가 Case에 없습니다")

            logging.info(f"ROI 매핑 딕셔너리: {roi_mapping}")

            # BeamSet 정보
            beam_set_name = None
            if len(plan.BeamSets) > 0:
                beam_set_name = plan.BeamSets[0].DicomPlanLabel
                logging.info(f"BeamSet: {beam_set_name}")

            # Template 적용 - 딕셔너리 형식으로!
            try:
                with CompositeAction('Apply Clinical Goals Template'):
                    apply_params = {
                        'Template': template,
                        'AssociatedRoisAndPois': roi_mapping,  # 딕셔너리 형식!
                        'AddClinicalGoalsDefinedOnTotalDose': True,
                        'ReplaceExistingClinicalGoals': False
                    }

                    # BeamSet 매핑 추가 (있으면)
                    if beam_set_name:
                        apply_params['AssociatedBeamSets'] = {'Beamset1': beam_set_name}

                    logging.info(f"적용 파라미터: Template={template_name}, ROIs={len(roi_mapping)}개")

                    plan.TreatmentCourse.EvaluationSetup.ApplyClinicalGoalTemplate(**apply_params)

                logging.info(f"✓ Template '{template_name}' 적용 성공")

                # 적용된 Clinical Goals 확인
                applied_goals = list(plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions)
                logging.info(f"적용된 Clinical Goals: {len(applied_goals)}개")

                for i, goal in enumerate(applied_goals):
                    try:
                        if hasattr(goal, 'ForRegionOfInterest') and goal.ForRegionOfInterest:
                            roi_name = goal.ForRegionOfInterest.Name
                            goal_type = type(goal).__name__
                            logging.info(f"  Goal {i+1}: {roi_name} ({goal_type})")
                    except:
                        logging.info(f"  Goal {i+1}: (정보 추출 실패)")

                if len(applied_goals) > 0:
                    logging.info("✅ Clinical Goals Template 적용 완료")
                    return True
                else:
                    logging.warning("⚠️ Clinical Goals가 적용되지 않았습니다")

            except Exception as e:
                logging.error(f"Template 적용 실패: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())

            # 방법 2: Template 구조 분석 후 수동 적용 (기존 방식)
            # Template 로드
            logging.info(f"Loading clinical goals template: {template_name}")
            template = patient_db.LoadTemplateClinicalGoals(
                templateName=template_name,
                lockMode='Read'
            )

            if template is None:
                logging.error(f"❌ Template '{template_name}' 로드 실패")
                return False

            logging.info(f"✓ Template '{template_name}' 로드 성공")

            # Template 객체 구조 디버깅
            logging.info("=== Template 객체 구조 분석 ===")
            logging.info(f"Template type: {type(template)}")
            template_attrs = [attr for attr in dir(template) if not attr.startswith('_')]
            logging.info(f"Template attributes: {template_attrs[:20]}...")  # 처음 20개만

            # BeamSet 정보 가져오기
            beam_set = plan.BeamSets[0] if len(plan.BeamSets) > 0 else None

            # ROI 매핑 정보 생성 (자동 매핑)
            try:
                # Template의 ROI 목록 추출 - 여러 경로 시도
                template_rois = []

                # 방법 1: EvaluationSetups (일반적인 경로)
                if hasattr(template, 'EvaluationSetups') and template.EvaluationSetups:
                    logging.info("시도 1: EvaluationSetups 경로")
                    try:
                        eval_setups_list = list(template.EvaluationSetups)
                        logging.info(f"  EvaluationSetups 개수: {len(eval_setups_list)}")

                        for i, eval_setup in enumerate(eval_setups_list):
                            logging.info(f"  EvaluationSetup {i}: {type(eval_setup)}")
                            eval_setup_attrs = [attr for attr in dir(eval_setup) if not attr.startswith('_')]
                            logging.info(f"    속성들: {eval_setup_attrs[:15]}")

                            # EvaluationFunctions 확인
                            if hasattr(eval_setup, 'EvaluationFunctions'):
                                eval_funcs = list(eval_setup.EvaluationFunctions)
                                logging.info(f"    EvaluationFunctions 개수: {len(eval_funcs)}")

                                for j, func in enumerate(eval_funcs):
                                    logging.info(f"      Function {j}: {type(func)}")

                                    # Function의 속성 확인
                                    func_attrs = [attr for attr in dir(func) if not attr.startswith('_')]
                                    logging.info(f"        Function 속성: {func_attrs[:10]}")

                                    # ForRegionOfInterest 확인
                                    if hasattr(func, 'ForRegionOfInterest'):
                                        try:
                                            for_roi = func.ForRegionOfInterest
                                            logging.info(f"        ForRegionOfInterest: {for_roi}, type: {type(for_roi)}")

                                            if for_roi:
                                                if hasattr(for_roi, 'Name'):
                                                    roi_name = for_roi.Name
                                                    logging.info(f"        ✓ ROI 이름 찾음: {roi_name}")
                                                    if roi_name and roi_name not in template_rois:
                                                        template_rois.append(roi_name)
                                                else:
                                                    logging.warning(f"        ForRegionOfInterest에 Name 속성 없음")
                                        except Exception as roi_error:
                                            logging.error(f"        ROI 추출 오류: {str(roi_error)}")
                                    else:
                                        logging.warning(f"        Function {j}에 ForRegionOfInterest 속성 없음")

                                        # PlanningGoal 경로 시도
                                        if hasattr(func, 'PlanningGoal'):
                                            try:
                                                pg = func.PlanningGoal
                                                if hasattr(pg, 'ForRegionOfInterest') and pg.ForRegionOfInterest:
                                                    roi_name = pg.ForRegionOfInterest.Name
                                                    logging.info(f"        ✓ ROI 이름 찾음 (PlanningGoal 경로): {roi_name}")
                                                    if roi_name and roi_name not in template_rois:
                                                        template_rois.append(roi_name)
                                            except:
                                                pass
                            else:
                                logging.warning(f"    EvaluationSetup {i}에 EvaluationFunctions 속성 없음")

                    except Exception as e:
                        logging.error(f"  EvaluationSetups 처리 중 오류: {str(e)}")
                        import traceback
                        logging.error(traceback.format_exc())

                # 방법 2: ClinicalGoals 직접 접근
                if len(template_rois) == 0 and hasattr(template, 'ClinicalGoals'):
                    logging.info("시도 2: ClinicalGoals 경로")
                    for goal in template.ClinicalGoals:
                        if hasattr(goal, 'ForRegionOfInterest') and goal.ForRegionOfInterest:
                            roi_name = goal.ForRegionOfInterest.Name
                            if roi_name and roi_name not in template_rois:
                                template_rois.append(roi_name)

                # 방법 3: Goals 직접 접근
                if len(template_rois) == 0 and hasattr(template, 'Goals'):
                    logging.info("시도 3: Goals 경로")
                    for goal in template.Goals:
                        if hasattr(goal, 'ForRegionOfInterest') and goal.ForRegionOfInterest:
                            roi_name = goal.ForRegionOfInterest.Name
                            if roi_name and roi_name not in template_rois:
                                template_rois.append(roi_name)

                # 방법 4: FunctionToRoiMaps 확인 (가장 유력한 경로!)
                if len(template_rois) == 0 and hasattr(template, 'FunctionToRoiMaps'):
                    logging.info("시도 4: FunctionToRoiMaps 경로")
                    try:
                        roi_maps = list(template.FunctionToRoiMaps)
                        logging.info(f"  FunctionToRoiMaps 개수: {len(roi_maps)}")

                        for i, roi_map in enumerate(roi_maps):
                            logging.info(f"    Map {i}: {type(roi_map)}")

                            # roi_map의 모든 속성 확인
                            map_attrs = [attr for attr in dir(roi_map) if not attr.startswith('_')]
                            logging.info(f"      속성: {map_attrs}")

                            # RtpFunction 확인 (ROI 정보가 여기에 있을 것!)
                            try:
                                if hasattr(roi_map, 'RtpFunction'):
                                    rtp_func = roi_map.RtpFunction
                                    logging.info(f"        RtpFunction type: {type(rtp_func)}")

                                    if rtp_func is not None:
                                        # RtpFunction의 속성 확인
                                        func_attrs = [attr for attr in dir(rtp_func) if not attr.startswith('_')]
                                        logging.info(f"        RtpFunction 속성: {func_attrs[:15]}")

                                        # 가능한 ROI 이름 속성들 확인
                                        for attr in ['ForRegionOfInterest', 'OfRoi', 'RoiName', 'Name']:
                                            if hasattr(rtp_func, attr):
                                                try:
                                                    value = getattr(rtp_func, attr)
                                                    logging.info(f"          {attr} = {value} (type: {type(value).__name__})")

                                                    # ROI 이름 추출
                                                    if isinstance(value, str) and value:
                                                        roi_name = value
                                                        logging.info(f"      ✓ ROI 찾음 (RtpFunction.{attr}): {roi_name}")
                                                        if roi_name not in template_rois:
                                                            template_rois.append(roi_name)
                                                        break
                                                    elif value is not None and hasattr(value, 'Name'):
                                                        roi_name = value.Name
                                                        logging.info(f"      ✓ ROI 찾음 (RtpFunction.{attr}.Name): {roi_name}")
                                                        if roi_name not in template_rois:
                                                            template_rois.append(roi_name)
                                                        break
                                                except Exception as e:
                                                    logging.warning(f"          {attr} 접근 실패: {str(e)}")
                                    else:
                                        logging.warning(f"        RtpFunction is None")
                            except Exception as e:
                                logging.error(f"        RtpFunction 처리 중 오류: {str(e)}")
                    except Exception as e:
                        logging.error(f"  FunctionToRoiMaps 처리 중 오류: {str(e)}")

                # 방법 5: 속성 탐색
                if len(template_rois) == 0:
                    logging.info("시도 5: 모든 속성 탐색")
                    for attr_name in template_attrs[:30]:  # 처음 30개 속성 확인
                        try:
                            attr_value = getattr(template, attr_name)
                            if hasattr(attr_value, '__iter__') and not isinstance(attr_value, str):
                                logging.info(f"  - {attr_name}: {type(attr_value)}, len={len(list(attr_value)) if hasattr(attr_value, '__len__') else 'N/A'}")
                        except:
                            pass

                logging.info(f"Template에서 추출한 ROIs ({len(template_rois)}개): {template_rois}")

                # Template이 비어있는 경우 처리
                if len(template_rois) == 0:
                    logging.error("❌ Template 'b_dm_g'가 비어있습니다!")
                    logging.error("Template에 Clinical Goals가 정의되어 있지 않습니다.")

                    # 사용자에게 확인
                    result = messagebox.askyesno(
                        "Template 비어있음",
                        f"Template 'b_dm_g'에 Clinical Goals가 없습니다.\n\n"
                        "가능한 해결책:\n"
                        "1. RayStation에서 template 'b_dm_g'를 확인\n"
                        "2. 올바른 template 이름 확인\n"
                        "3. 수동으로 Clinical Goals 설정\n\n"
                        "수동으로 Clinical Goals를 설정하고 계속하시겠습니까?"
                    )

                    if not result:
                        logging.info("사용자가 스크립트를 취소했습니다.")
                        return False

                    logging.info("✅ 사용자가 수동 설정을 선택했습니다. 계속 진행합니다.")
                    return True

                # 현재 Plan의 ROI 목록
                patient_rois = [roi.Name for roi in case.PatientModel.RegionsOfInterest]
                logging.info(f"Patient ROIs ({len(patient_rois)}개): {patient_rois}")

                # 자동 ROI 매핑 생성
                roi_mapping = {}
                for template_roi in template_rois:
                    # 정확히 일치하는 ROI 찾기
                    if template_roi in patient_rois:
                        roi_mapping[template_roi] = template_roi
                        logging.info(f"  ✓ 자동 매핑: {template_roi} → {template_roi}")

                if roi_mapping:
                    logging.info(f"총 {len(roi_mapping)}개 ROI 자동 매핑 완료")
                else:
                    logging.warning("⚠️ 자동 매핑된 ROI가 0개입니다!")
                    logging.warning(f"Template ROIs: {template_rois}")
                    logging.warning(f"Patient ROIs: {patient_rois}")

            except Exception as mapping_error:
                logging.error(f"ROI 매핑 생성 중 오류: {str(mapping_error)}")
                import traceback
                logging.error(traceback.format_exc())
                roi_mapping = {}

            # Template 적용 - 기존 goals 유지하면서 추가
            with CompositeAction('Apply Clinical Goals Template'):
                try:
                    # ROI 매핑과 함께 적용
                    apply_params = {
                        'Template': template,
                        'ReplaceExistingClinicalGoals': False
                    }

                    if beam_set is not None:
                        apply_params['AssociatedBeamSets'] = [beam_set]

                    # ROI 매핑 추가 (파라미터 이름 확인 필요)
                    if roi_mapping:
                        try:
                            # RayStation API에서 ROI 매핑 파라미터 시도
                            apply_params['AssociatedRoisAndPois'] = roi_mapping
                        except:
                            pass

                    plan.TreatmentCourse.EvaluationSetup.ApplyClinicalGoalTemplate(**apply_params)
                    logging.info("✓ Template 적용 완료 (ROI 매핑 포함)")

                except Exception as e:
                    logging.warning(f"Template 적용 시도 1 실패: {str(e)}")
                    # 방법 2: 최소 파라미터로 재시도
                    try:
                        plan.TreatmentCourse.EvaluationSetup.ApplyClinicalGoalTemplate(
                            Template=template,
                            ReplaceExistingClinicalGoals=False
                        )
                        logging.info("✓ Template 적용 (최소 파라미터)")
                    except Exception as e2:
                        logging.error(f"Template 적용 실패: {str(e2)}")
                        raise

            # 로드된 goals 개수 확인
            loaded_goals = list(plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions)
            new_goals_count = len(loaded_goals) - len(existing_goals)

            logging.info(f"✅ Template '{template_name}' 적용 완료")
            logging.info(f"   기존 Clinical Goals: {len(existing_goals)}개")
            logging.info(f"   적용 후 Clinical Goals: {len(loaded_goals)}개")
            logging.info(f"   추가된 Clinical Goals: {new_goals_count}개")

            if new_goals_count == 0:
                logging.warning("⚠️ Template에서 추가된 Clinical Goals가 0개입니다.")
                logging.warning("⚠️ 가능한 원인:")
                logging.warning("   1. Template의 ROI 이름과 현재 Plan의 ROI 이름이 다름")
                logging.warning("   2. Template이 비어있음")
                logging.warning("   3. BeamSet 연결 문제")

                # Plan의 ROI 목록 출력
                try:
                    case = get_current("Case")
                    roi_names = [roi.Name for roi in case.PatientModel.RegionsOfInterest]
                    logging.info(f"현재 Plan의 ROI 목록 ({len(roi_names)}개): {', '.join(roi_names[:15])}...")
                except:
                    pass

                # 사용자에게 확인 요청
                result = messagebox.askyesno(
                    "Clinical Goals 확인",
                    f"Template '{template_name}'에서 추가된 Clinical Goals가 0개입니다.\n\n"
                    "ROI 이름 불일치 또는 Template이 비어있을 수 있습니다.\n\n"
                    "계속 진행하시겠습니까?\n\n"
                    "'Yes': 스크립트 계속\n"
                    "'No': 스크립트 종료"
                )

                if not result:
                    logging.info("사용자가 스크립트를 취소했습니다.")
                    return False

            return True

        except Exception as e:
            logging.error(f"❌ Template '{template_name}' 로드 중 오류: {str(e)}")

            # 사용자에게 수동 설정 안내
            result = messagebox.askyesno(
                "Clinical Goals Template 오류",
                f"Template 로드 중 오류:\n{str(e)}\n\n"
                "수동으로 Clinical Goals를 설정하시겠습니까?\n\n"
                "'Yes'를 선택하면 스크립트를 계속 진행합니다."
            )

            if not result:
                return False

            logging.info("✅ Clinical Goals 확인 완료 (사용자 설정)")
            return True

    except Exception as e:
        logging.error(f"Clinical Goal Template 로딩 중 치명적 오류: {str(e)}")
        return False


def load_clinical_goals_safe(plan):
    """Clinical Goals를 안전하게 로드"""
    try:
        evaluation_functions = plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions
        goals = []

        for func in evaluation_functions:
            try:
                if hasattr(func, 'PlanningGoal') or hasattr(func, 'ForRegionOfInterest'):
                    goals.append(func)
            except Exception as e:
                logging.debug(f"Goal 추가 중 오류 (무시): {str(e)}")
                continue

        logging.info(f"총 {len(goals)}개의 Clinical Goals를 안전하게 로드함")
        return goals

    except Exception as e:
        logging.error(f"Clinical Goals 로드 실패: {str(e)}")
        return []



def get_clinical_goals_safely(plan):
    """
    Clinical Goals를 안전하게 가져오기 (별칭 함수)
    main()에서 호출되는 함수
    """
    return load_clinical_goals_safe(plan)

def check_selected_goals(plan, selected_goals):
    """선택된 목표들이 모두 달성되었는지 확인"""
    try:
        all_met = True
        met_count = 0

        for goal in selected_goals:
            try:
                planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)

                current_value = None
                try:
                    current_value = goal.GetClinicalGoalValue()
                except:
                    pass

                is_met = False
                if current_value is not None and acceptance_level > 0:
                    try:
                        if goal_criteria == "AtMost":
                            is_met = current_value <= acceptance_level
                        elif goal_criteria == "AtLeast":
                            is_met = current_value >= acceptance_level
                    except:
                        pass

                if is_met:
                    met_count += 1
                else:
                    all_met = False

            except Exception as e:
                logging.debug(f"Goal 체크 중 오류 (무시): {str(e)}")
                all_met = False

        logging.info(f"목표 달성 현황: {met_count}/{len(selected_goals)}")
        return all_met, met_count

    except Exception as e:
        logging.error(f"check_selected_goals 오류: {str(e)}")
        return False, 0


def get_number_of_fractions(beam_set):
    """BeamSet의 fraction 수 가져오기"""
    try:
        return beam_set.FractionationPattern.NumberOfFractions
    except:
        # NumberOfFractions를 가져올 수 없으면 기본값
        logging.warning("NumberOfFractions를 가져올 수 없음 - 기본값 5 사용")
        return 5


def get_ctv_wb_max_dose(plan):
    """CTV_WB의 최대 선량을 가져오기 (Total Dose)"""
    try:
        beam_set = plan.BeamSets[0]
        roi_name = "CTV_WB"

        # Fraction 수 가져오기
        num_fractions = get_number_of_fractions(beam_set)

        # FractionDose에서 최대 선량 가져오기
        dose_distribution = beam_set.FractionDose
        max_dose_per_fraction = dose_distribution.GetDoseStatistic(RoiName=roi_name, DoseType='Max')

        # Total Dose 계산
        max_dose_total = max_dose_per_fraction * num_fractions

        logging.debug(f"Max Dose: {max_dose_per_fraction:.1f} cGy/fx × {num_fractions} fx = {max_dose_total:.1f} cGy")

        return max_dose_total

    except Exception as e:
        logging.error(f"CTV_WB 최대 선량 가져오기 실패: {str(e)}")
        return 0


def get_ctv_wb_d95_total(beam_set):
    """CTV_WB D95% 가져오기 (Total Dose)"""
    try:
        num_fractions = get_number_of_fractions(beam_set)

        d95_per_fraction = beam_set.FractionDose.GetDoseAtRelativeVolumes(
            RoiName="CTV_WB",
            RelativeVolumes=[0.95]
        )[0]

        d95_total = d95_per_fraction * num_fractions

        logging.debug(f"D95%: {d95_per_fraction:.1f} cGy/fx × {num_fractions} fx = {d95_total:.1f} cGy")

        return d95_total

    except Exception as e:
        logging.error(f"CTV_WB D95% 가져오기 실패: {str(e)}")
        return 0


def calculate_smart_prescription(current_d95, current_max, selected_goals):
    """
    MinDVH-MaxDose heterogeneity를 고려한 최적 prescription 계산

    Args:
        current_d95: 현재 D95% 값 (cGy)
        current_max: 현재 Max Dose 값 (cGy)
        selected_goals: Clinical goals 리스트

    Returns:
        (feasible, optimal_prescription, reason):
            - feasible: True if feasible range exists
            - optimal_prescription: Optimal prescription dose (cGy)
            - reason: Explanation string
    """
    try:
        # 1. Goal 정보 추출
        d95_goal = None
        max_goal = None

        for goal_info in selected_goals:
            # Dictionary 또는 RayStation 객체 모두 지원
            if isinstance(goal_info, dict):
                roi_name = goal_info.get('RoiName', '')
                goal_type = goal_info.get('Type', '')
                goal_criteria = goal_info.get('Criteria', '')
                acceptance = goal_info.get('AcceptanceLevel', 0)
            else:
                # RayStation 객체인 경우
                try:
                    roi_name = goal_info.ForRegionOfInterest.Name if hasattr(goal_info, 'ForRegionOfInterest') else ''
                    planning_goal = goal_info.PlanningGoal if hasattr(goal_info, 'PlanningGoal') else None
                    if planning_goal:
                        goal_type = getattr(planning_goal, 'Type', '')
                        goal_criteria = getattr(planning_goal, 'GoalCriteria', '')
                        acceptance = getattr(planning_goal, 'PrimaryAcceptanceLevel', 0)
                    else:
                        continue
                except Exception as e:
                    logging.debug(f"Goal 정보 추출 실패: {str(e)}")
                    continue

            if roi_name == "CTV_WB":
                if goal_type == "DoseAtVolume" and goal_criteria == "AtLeast":
                    d95_goal = acceptance  # 2470 cGy
                elif goal_type == "DoseAtAbsoluteVolume" and goal_criteria == "AtMost":
                    max_goal = acceptance  # 2781 cGy

        if not d95_goal or not max_goal:
            return False, 0, "D95% 또는 MaxDose goal을 찾을 수 없음"

        logging.info(f"=== Smart Prescription 계산 ===")
        logging.info(f"D95% Goal: >= {d95_goal} cGy")
        logging.info(f"MaxDose Goal: <= {max_goal} cGy")

        # 2. 현재 dose heterogeneity 계산
        if current_d95 <= 0:
            return False, 0, "D95% 값이 유효하지 않음"

        heterogeneity = current_max / current_d95
        logging.info(f"현재 Heterogeneity (Max/D95): {heterogeneity:.4f}")
        logging.info(f"  → Max = D95 × {heterogeneity:.4f}")

        # 3. Feasible prescription range 계산
        # 하한: D95% >= 2470을 만족하려면
        # D95 >= 2470 → prescription × 0.95 >= 2470
        min_prescription = d95_goal / 0.95

        # 상한: MaxDose <= 2781을 만족하려면
        # Max = D95 × heterogeneity <= 2781
        # D95 <= 2781 / heterogeneity
        # prescription × 0.95 <= 2781 / heterogeneity
        # prescription <= (2781 / heterogeneity) / 0.95
        max_prescription = (max_goal / heterogeneity) / 0.95

        logging.info(f"Feasible Prescription Range:")
        logging.info(f"  하한 (D95% goal 만족): >= {min_prescription:.1f} cGy")
        logging.info(f"  상한 (MaxDose goal 만족): <= {max_prescription:.1f} cGy")

        # 4. Feasibility 확인
        if min_prescription > max_prescription:
            gap = min_prescription - max_prescription
            reason = (
                f"Infeasible! Gap = {gap:.1f} cGy\n"
                f"  D95% goal을 만족하려면 >= {min_prescription:.1f} cGy 필요\n"
                f"  MaxDose goal을 만족하려면 <= {max_prescription:.1f} cGy 필요\n"
                f"  → Heterogeneity가 너무 높음 (추가 최적화 필요)"
            )
            logging.warning(reason)
            return False, 0, reason

        # 5. 최적 prescription 선택 (안전 마진 고려)
        # 중간값 선택하되, 약간 상한 쪽으로 치우침 (D95% 여유 확보)
        safety_bias = 0.6  # 60% 상한 쪽
        optimal_prescription = min_prescription + (max_prescription - min_prescription) * safety_bias

        # 6. 예상 결과 계산
        predicted_d95 = optimal_prescription * 0.95
        predicted_max = predicted_d95 * heterogeneity

        reason = (
            f"Feasible range: {max_prescription - min_prescription:.1f} cGy\n"
            f"  최적 Prescription: {optimal_prescription:.1f} cGy\n"
            f"  예상 D95%: {predicted_d95:.1f} cGy (goal: >= {d95_goal})\n"
            f"  예상 MaxDose: {predicted_max:.1f} cGy (goal: <= {max_goal})"
        )
        logging.info(f"✅ Feasible! {reason}")

        return True, optimal_prescription, reason

    except Exception as e:
        logging.error(f"Smart Prescription 계산 실패: {str(e)}")
        return False, 0, str(e)


def try_prescription_scaling_for_unmet_goals(plan, beam_set, selected_goals, current_d95, current_max):
    """
    미달성 goal이 있을 때, prescription scaling으로 모든 goal 달성 가능한지 시도
    Smart Scaling: MinDVH-MaxDose heterogeneity 고려

    Args:
        plan: RayStation plan object
        beam_set: Beam set object
        selected_goals: 선택된 clinical goals 리스트
        current_d95: 현재 CTV_WB D95% 값 (cGy)
        current_max: 현재 CTV_WB Max Dose 값 (cGy)

    Returns:
        (scaling_success, final_max, final_d95):
            - scaling_success: True if all goals met after scaling
            - final_max: Final max dose after scaling
            - final_d95: Final D95% after scaling
    """
    try:
        logging.info(f"현재 상태: D95% = {current_d95:.1f} cGy, Max = {current_max:.1f} cGy")

        # 1. 현재 prescription 정보 가져오기
        try:
            current_prescription = beam_set.Prescription.PrimaryPrescriptionDoseReference
            current_total_dose = current_prescription.DoseValue
            logging.info(f"현재 Prescription: {current_total_dose} cGy at 95%")
        except:
            logging.error("Prescription 정보를 가져올 수 없습니다.")
            return False, current_max, current_d95

        # 2. Smart Prescription 계산 (Heterogeneity 고려)
        feasible, new_total_dose, reason = calculate_smart_prescription(
            current_d95, current_max, selected_goals
        )

        if not feasible:
            logging.warning("Smart Prescription Scaling 불가능:")
            logging.warning(reason)
            logging.info("추가 최적화 cycle이 필요합니다.")
            return False, current_max, current_d95

        logging.info(f"Smart Prescription: {new_total_dose:.1f} cGy")
        scale_factor = new_total_dose / current_total_dose
        logging.info(f"Scale factor: {scale_factor:.4f}")

        # 3. 예상 결과
        scaled_d95 = new_total_dose * 0.95
        scaled_max = current_max * scale_factor

        logging.info(f"Scaled 예상값: D95% = {scaled_d95:.1f} cGy, Max = {scaled_max:.1f} cGy")

        # 4. Prescription 업데이트하고 실제로 goal 재평가
        logging.info(f"Prescription을 {current_total_dose:.1f} → {new_total_dose:.1f} cGy로 업데이트 시도...")

        try:
            with CompositeAction('Update Prescription via Scaling'):
                current_prescription.DoseValue = new_total_dose
                beam_set.ComputeDose(
                    ComputeBeamDoses=True,
                    DoseAlgorithm="CCDose",
                    ForceRecompute=True
                )

            # 5. 업데이트 후 실제 값 확인 (Total Dose)
            final_d95 = get_ctv_wb_d95_total(beam_set)
            final_max = get_ctv_wb_max_dose(plan)

            logging.info(f"Prescription 업데이트 후 D95% (Total): {final_d95:.1f} cGy")
            logging.info(f"Prescription 업데이트 후 Max (Total): {final_max:.1f} cGy")

            # 6. 모든 goal이 달성되었는지 확인
            met_count = 0
            total_goals = len(selected_goals)
            unmet_goals = []

            for goal in selected_goals:
                try:
                    planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                    goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                    acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)

                    current_value = None
                    try:
                        current_value = goal.GetClinicalGoalValue()
                    except:
                        pass

                    goal_met = False
                    if current_value is not None and acceptance_level > 0:
                        try:
                            if goal_criteria == "AtMost":
                                goal_met = current_value <= acceptance_level
                            elif goal_criteria == "AtLeast":
                                goal_met = current_value >= acceptance_level
                        except:
                            pass

                    if goal_met:
                        met_count += 1
                    else:
                        roi_name = goal.ForRegionOfInterest.Name if hasattr(goal, 'ForRegionOfInterest') else 'Unknown'
                        goal_type = getattr(planning_goal, 'Type', 'Unknown')
                        unmet_goals.append(f"{roi_name} - {goal_type}")

                except Exception as e:
                    logging.warning(f"Goal 평가 중 오류: {str(e)}")

            logging.info(f"Prescription Scaling 후 달성된 Goals: {met_count}/{total_goals}")

            if met_count == total_goals:
                logging.info(f"✅ Prescription Scaling으로 모든 Goals 달성!")
                logging.info(f"최종 Prescription: {new_total_dose:.1f} cGy")
                logging.info(f"최종 D95%: {final_d95:.1f} cGy")
                logging.info(f"최종 Max: {final_max:.1f} cGy")
                plan.Save()
                return True, final_max, final_d95
            else:
                logging.warning(f"Prescription Scaling 후에도 {total_goals - met_count}개 goals 미달성:")
                for goal in unmet_goals:
                    logging.warning(f"  - {goal}")
                logging.info("Prescription을 원래대로 복구합니다.")

                # Prescription 복구
                with CompositeAction('Restore Original Prescription'):
                    current_prescription.DoseValue = current_total_dose
                    beam_set.ComputeDose(
                        ComputeBeamDoses=True,
                        DoseAlgorithm="CCDose",
                        ForceRecompute=True
                    )

                return False, current_max, current_d95

        except Exception as e:
            logging.error(f"Prescription 업데이트 실패: {str(e)}")
            return False, current_max, current_d95

    except Exception as e:
        logging.error(f"Prescription Scaling 시도 중 오류: {str(e)}")
        return False, current_max, current_d95


def apply_ctv_wb_fallback_objectives(plan, beam_set):
    """
    CTV_WB 최적화 실패 시 fallback 로직:
    Dose threshold 기반 ROI를 생성하고 MaxDose와 MinDVH objective 추가

    추가 조건:
    1. Subtraction: 2500cGy threshold → CTV_WB와 교집합 → CTV_WB - 교집합
       → MinDVH 96% at 2550cGy with weight 100000
    2. MaxDose: 2700cGy with weight 1000000
    3. MaxDose: 2710cGy with weight 1000000

    Returns:
        bool: 성공 여부
    """
    try:
        logging.info("\n" + "="*60)
        logging.info("⚠️ CTV_WB Fallback 로직 실행")
        logging.info("Dose threshold 기반 ROI 생성 및 Objective 추가")
        logging.info("="*60)

        case = get_current("Case")
        patient = get_current("Patient")
        examination = get_current("Examination")
        plan_optimization = plan.PlanOptimizations[0]

        # Total dose 가져오기
        try:
            plan_dose = plan.TreatmentCourse.TotalDose
        except Exception as e:
            logging.error(f"Total dose 접근 실패: {str(e)}")
            return False

        # CTV_WB 존재 확인
        ctv_wb_name = None
        try:
            for roi in case.PatientModel.RegionsOfInterest:
                if roi.Name.upper() == "CTV_WB":
                    ctv_wb_name = roi.Name
                    break
        except Exception as e:
            logging.error(f"CTV_WB ROI 검색 실패: {str(e)}")
            return False

        if not ctv_wb_name:
            logging.error("CTV_WB ROI를 찾을 수 없습니다.")
            return False

        logging.info(f"CTV_WB ROI 발견: {ctv_wb_name}")

        # Helper function: ROI 이름 생성
        def get_unique_roi_name(prefix):
            try:
                existing_rois = [r.Name for r in case.PatientModel.RegionsOfInterest if r.Name.startswith(prefix)]
            except:
                existing_rois = []

            used_indices = []
            for roi_name in existing_rois:
                suffix = roi_name[len(prefix):]
                if suffix.isdigit():
                    used_indices.append(int(suffix))

            if used_indices:
                next_index = max(used_indices) + 1
            else:
                next_index = 1

            return f"{prefix}{next_index}"

        # Helper function: Dose threshold ROI 생성
        def create_dose_threshold_roi(dose_cgy, roi_name, color="Gray"):
            try:
                new_roi = case.PatientModel.CreateRoi(
                    Name=roi_name,
                    Color=color,
                    Type="Control",
                    TissueName=None,
                    RbeCellTypeName=None,
                    RoiMaterial=None
                )

                new_roi.CreateRoiGeometryFromDose(
                    DoseDistribution=plan_dose,
                    ThresholdLevel=dose_cgy
                )

                # ExcludeFromExport 설정
                try:
                    case.PatientModel.ToggleExcludeFromExport(
                        ExcludeFromExport=True,
                        RegionOfInterests=[roi_name],
                        PointsOfInterests=[]
                    )
                except:
                    pass

                # Volume 확인
                try:
                    beam_set.FractionDose.UpdateDoseGridStructures()
                except:
                    pass

                patient.Save()

                roi_geo = case.PatientModel.StructureSets[examination.Name].RoiGeometries[roi_name]
                if not roi_geo.HasContours():
                    return None, 0.0

                volume = roi_geo.GetRoiVolume()
                return new_roi, volume

            except Exception as e:
                logging.error(f"ROI 생성 실패 ({roi_name}): {str(e)}")
                return None, 0.0

        # Helper function: ROI 교집합 및 빼기 연산 (Subtraction)
        def apply_complex_subtraction(target_roi_name, base_roi_name):
            try:
                margin_settings = {
                    'Type': "Expand",
                    'Superior': 0, 'Inferior': 0,
                    'Anterior': 0, 'Posterior': 0,
                    'Right': 0, 'Left': 0
                }

                # 1단계: 교집합
                with CompositeAction(f'ROI algebra intersection ({target_roi_name})'):
                    case.PatientModel.RegionsOfInterest[target_roi_name].CreateAlgebraGeometry(
                        Examination=examination,
                        Algorithm="Auto",
                        ExpressionA={'Operation': "Union", 'SourceRoiNames': [target_roi_name], 'MarginSettings': margin_settings},
                        ExpressionB={'Operation': "Union", 'SourceRoiNames': [base_roi_name], 'MarginSettings': margin_settings},
                        ResultOperation="Intersection",
                        ResultMarginSettings=margin_settings
                    )

                # 2단계: 빼기
                with CompositeAction(f'ROI algebra subtraction ({target_roi_name})'):
                    case.PatientModel.RegionsOfInterest[target_roi_name].CreateAlgebraGeometry(
                        Examination=examination,
                        Algorithm="Auto",
                        ExpressionA={'Operation': "Union", 'SourceRoiNames': [base_roi_name], 'MarginSettings': margin_settings},
                        ExpressionB={'Operation': "Union", 'SourceRoiNames': [target_roi_name], 'MarginSettings': margin_settings},
                        ResultOperation="Subtraction",
                        ResultMarginSettings=margin_settings
                    )

                try:
                    beam_set.FractionDose.UpdateDoseGridStructures()
                except:
                    pass

                patient.Save()
                return True

            except Exception as e:
                logging.error(f"Subtraction 연산 실패: {str(e)}")
                return False

        success_count = 0

        # 1. Subtraction ROI: 2500cGy threshold → MinDVH 96% at 2550cGy
        logging.info("\n[1/3] Subtraction ROI 생성: 2500cGy threshold")
        sub_roi_name = get_unique_roi_name(f"3_Sub_2500_{ctv_wb_name}_")
        sub_roi, sub_volume = create_dose_threshold_roi(2500, sub_roi_name, color="Orange")

        if sub_roi and sub_volume > 0:
            logging.info(f"✓ Subtraction ROI 생성 완료: {sub_roi_name} (Volume: {sub_volume:.2f} cc)")

            # Subtraction 연산 수행
            if apply_complex_subtraction(sub_roi_name, ctv_wb_name):
                # 최종 volume 확인
                final_geo = case.PatientModel.StructureSets[examination.Name].RoiGeometries[sub_roi_name]
                if final_geo.HasContours():
                    final_volume = final_geo.GetRoiVolume()

                    if final_volume > 0:
                        logging.info(f"✓ Subtraction 연산 완료: {sub_roi_name} (최종 Volume: {final_volume:.2f} cc)")

                        # MinDVH Objective 추가
                        try:
                            func = plan_optimization.AddOptimizationFunction(
                                FunctionType="MinDvh",
                                RoiName=sub_roi_name,
                                IsConstraint=False,
                                RestrictAllBeamsIndividually=False,
                                IsRobust=False,
                                RestrictToBeamSet=None,
                                UseRbeDose=False
                            )
                            func.DoseFunctionParameters.DoseLevel = 2550
                            func.DoseFunctionParameters.PercentVolume = 96
                            func.DoseFunctionParameters.Weight = 100000

                            logging.info(f"✓ MinDVH Objective 추가: {sub_roi_name}, DoseLevel=2550cGy, Volume>=96%, Weight=100000")
                            success_count += 1

                        except Exception as e:
                            logging.error(f"MinDVH Objective 추가 실패: {str(e)}")
                    else:
                        logging.warning(f"Subtraction 결과 volume=0, ROI 삭제: {sub_roi_name}")
                        try:
                            case.PatientModel.RegionsOfInterest[sub_roi_name].DeleteRoi()
                        except:
                            pass
                else:
                    logging.warning(f"Subtraction 결과 contour 없음, ROI 삭제: {sub_roi_name}")
                    try:
                        case.PatientModel.RegionsOfInterest[sub_roi_name].DeleteRoi()
                    except:
                        pass
        else:
            logging.warning(f"Subtraction ROI 생성 실패 또는 volume=0")

        # 2. MaxDose ROI: 2700cGy
        logging.info("\n[2/3] MaxDose ROI 생성: 2700cGy")
        max1_roi_name = get_unique_roi_name("1_Max_2700_")
        max1_roi, max1_volume = create_dose_threshold_roi(2700, max1_roi_name, color="Red")

        if max1_roi and max1_volume > 0:
            logging.info(f"✓ MaxDose ROI 생성 완료: {max1_roi_name} (Volume: {max1_volume:.2f} cc)")

            # MaxDose Objective 추가
            try:
                func = plan_optimization.AddOptimizationFunction(
                    FunctionType="MaxDose",
                    RoiName=max1_roi_name,
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                func.DoseFunctionParameters.DoseLevel = 2700
                func.DoseFunctionParameters.Weight = 1000000

                logging.info(f"✓ MaxDose Objective 추가: {max1_roi_name}, DoseLevel=2700cGy, Weight=1000000")
                success_count += 1

            except Exception as e:
                logging.error(f"MaxDose Objective 추가 실패: {str(e)}")
                try:
                    case.PatientModel.RegionsOfInterest[max1_roi_name].DeleteRoi()
                except:
                    pass
        else:
            logging.warning(f"MaxDose ROI (2700cGy) 생성 실패 또는 volume=0")

        # 3. MaxDose ROI: 2710cGy
        logging.info("\n[3/3] MaxDose ROI 생성: 2710cGy")
        max2_roi_name = get_unique_roi_name("1_Max_2710_")
        max2_roi, max2_volume = create_dose_threshold_roi(2710, max2_roi_name, color="DarkRed")

        if max2_roi and max2_volume > 0:
            logging.info(f"✓ MaxDose ROI 생성 완료: {max2_roi_name} (Volume: {max2_volume:.2f} cc)")

            # MaxDose Objective 추가
            try:
                func = plan_optimization.AddOptimizationFunction(
                    FunctionType="MaxDose",
                    RoiName=max2_roi_name,
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                func.DoseFunctionParameters.DoseLevel = 2710
                func.DoseFunctionParameters.Weight = 1000000

                logging.info(f"✓ MaxDose Objective 추가: {max2_roi_name}, DoseLevel=2710cGy, Weight=1000000")
                success_count += 1

            except Exception as e:
                logging.error(f"MaxDose Objective 추가 실패: {str(e)}")
                try:
                    case.PatientModel.RegionsOfInterest[max2_roi_name].DeleteRoi()
                except:
                    pass
        else:
            logging.warning(f"MaxDose ROI (2710cGy) 생성 실패 또는 volume=0")

        # 최종 저장
        patient.Save()

        logging.info("\n" + "="*60)
        logging.info(f"✅ CTV_WB Fallback 완료: {success_count}/3 objectives 추가됨")
        logging.info("="*60)

        return success_count > 0

    except Exception as e:
        logging.error(f"CTV_WB Fallback 실패: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def run_scaling_optimization(plan, beam_set, selected_goals, status_gui):
    """스케일링 최적화 실행"""
    try:
        logging.info("=== 스케일링 최적화 시작 ===")

        target_d95 = 2471
        max_dose_limit = 2600
        lower_limit = 2460

        scale_factor = 1.0
        iteration = 0
        max_iterations = 20

        while iteration < max_iterations:
            iteration += 1

            # Total Dose로 가져오기
            current_d95 = get_ctv_wb_d95_total(beam_set)
            current_max = get_ctv_wb_max_dose(plan)

            logging.info(f"스케일링 반복 {iteration}: D95={current_d95:.1f}, Max={current_max:.1f}, Factor={scale_factor:.4f}")

            if target_d95 - 5 <= current_d95 <= target_d95 + 5 and current_max < max_dose_limit:
                logging.info(f"스케일링 목표 달성! D95={current_d95:.1f}, Max={current_max:.1f}")

                if status_gui:
                    met_count = check_selected_goals(plan, selected_goals)[1]
                    status_gui.update_status(
                        0, 0,
                        met_count, len(selected_goals),
                        current_max,
                        f"스케일링 완료 (반복 {iteration}회)"
                    )

                return True, current_max, current_d95

            if current_d95 < lower_limit:
                logging.warning(f"D95가 하방 제한({lower_limit}cGy)에 도달했습니다. 스케일링 중단.")

                if status_gui:
                    met_count = check_selected_goals(plan, selected_goals)[1]
                    status_gui.update_status(
                        0, 0,
                        met_count, len(selected_goals),
                        current_max,
                        f"스케일링 한계 도달 (D95={current_d95:.1f})"
                    )

                return False, current_max, current_d95

            if current_d95 < target_d95:
                scale_factor *= (target_d95 / current_d95)
            else:
                scale_factor *= 0.99

            beam_set.NormalizeToPrescription(
                RoiName="CTV_WB",
                DoseValue=target_d95 * scale_factor,
                DoseVolume=95,
                PrescriptionType="DoseAtVolume",
                LockedBeamNames=None,
                EvaluateAfterScaling=True
            )

            if status_gui:
                met_count = check_selected_goals(plan, selected_goals)[1]
                status_gui.update_status(
                    iteration, max_iterations,
                    met_count, len(selected_goals),
                    current_max,
                    f"스케일링 중 ({iteration}/{max_iterations})"
                )

        logging.warning("스케일링 최대 반복 횟수 도달")

        # Total Dose로 가져오기
        final_d95 = get_ctv_wb_d95_total(beam_set)
        final_max = get_ctv_wb_max_dose(plan)

        return False, final_max, final_d95

    except Exception as e:
        logging.error(f"스케일링 최적화 중 오류: {str(e)}")
        return False, 0, 0


def add_initial_objectives(plan):
    """
    초기 Objective Functions 추가:
    - 기존 Mimic dose objective 삭제
    - CTV_WB: MinDVH 2500cGy at 95% volume (weight 30000), MaxDose 2730cGy (weight 10000)
    - External: DoseFallOff 2600cGy at 1cm to 2300cGy (weight 1000)
    - Ring: MaxDose 2500cGy (weight 8000)

    Returns:
        bool: 성공 여부
    """
    try:
        logging.info("=== 초기 Objective Functions 추가 ===")

        plan_optimization = plan.PlanOptimizations[0]
        objectives_added = 0

        # 기존 Objective Functions 확인 및 삭제
        if hasattr(plan_optimization, 'Objective') and plan_optimization.Objective:
            existing_funcs = list(plan_optimization.Objective.ConstituentFunctions)
            logging.info(f"기존 Objective Functions: {len(existing_funcs)}개")

            # Mimic dose나 불필요한 objectives 삭제
            deleted_count = 0
            for func in existing_funcs:
                try:
                    func_type = func.DoseFunctionParameters.FunctionType
                    roi_name = func.ForRegionOfInterest.Name if func.ForRegionOfInterest else "Unknown"

                    # Mimic dose objective 삭제
                    if func_type in ["UniformDose", "TargetEud", "DoseFallOff"] and roi_name == "CTV_WB":
                        with CompositeAction(f'Delete {func_type} for {roi_name}'):
                            plan_optimization.Objective.ConstituentFunctions.Remove(func)
                        logging.info(f"  ✓ 기존 {func_type} objective 삭제: {roi_name}")
                        deleted_count += 1
                except Exception as e:
                    logging.debug(f"Objective 삭제 중 오류 (무시): {e}")

            if deleted_count > 0:
                logging.info(f"총 {deleted_count}개 기존 objective 삭제 완료")

        # 현재 Objective Functions 개수 확인
        initial_func_count = len(plan_optimization.Objective.ConstituentFunctions) if hasattr(plan_optimization, 'Objective') and plan_optimization.Objective else 0
        logging.info(f"현재 Objective Functions: {initial_func_count}개")

        # 1. CTV_WB MinDVH
        try:
            with CompositeAction('Add CTV_WB MinDVH'):
                plan_optimization.AddOptimizationFunction(
                    FunctionType="MinDvh",
                    RoiName="CTV_WB",
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    RestrictToBeams=[],
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                # 첫 번째 함수: 인덱스 = initial_func_count
                func_index = initial_func_count
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.DoseLevel = 2500
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.PercentVolume = 95
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.Weight = 100000

            logging.info("  ✓ CTV_WB MinDVH 2500cGy @ 95% (Weight: 100000)")
            objectives_added += 1
        except Exception as e:
            logging.error(f"  ✗ CTV_WB MinDVH 추가 실패: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

        # 2. CTV_WB MaxDose
        try:
            with CompositeAction('Add CTV_WB MaxDose'):
                plan_optimization.AddOptimizationFunction(
                    FunctionType="MaxDose",
                    RoiName="CTV_WB",
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    RestrictToBeams=[],
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                # 두 번째 함수: 인덱스 = initial_func_count + 1
                func_index = initial_func_count + 1
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.DoseLevel = 2730
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.Weight = 50000

            logging.info("  ✓ CTV_WB MaxDose 2730cGy (Weight: 50000)")
            objectives_added += 1
        except Exception as e:
            logging.error(f"  ✗ CTV_WB MaxDose 추가 실패: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

        # 3. External DoseFallOff
        try:
            with CompositeAction('Add External DoseFallOff'):
                plan_optimization.AddOptimizationFunction(
                    FunctionType="DoseFallOff",
                    RoiName="External",
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    RestrictToBeams=[],
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                # 세 번째 함수: 인덱스 = initial_func_count + 2
                func_index = initial_func_count + 2
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.HighDoseLevel = 2600
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.LowDoseLevel = 2300
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.Weight = 1000

            logging.info("  ✓ External DoseFallOff 2600cGy @ 1cm → 2300cGy (Weight: 1000)")
            objectives_added += 1
        except Exception as e:
            logging.error(f"  ✗ External DoseFallOff 추가 실패: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

        # 4. Ring MaxDose
        try:
            with CompositeAction('Add Ring MaxDose'):
                plan_optimization.AddOptimizationFunction(
                    FunctionType="MaxDose",
                    RoiName="RING",
                    IsConstraint=False,
                    RestrictAllBeamsIndividually=False,
                    RestrictToBeams=[],
                    IsRobust=False,
                    RestrictToBeamSet=None,
                    UseRbeDose=False
                )
                # 네 번째 함수: 인덱스 = initial_func_count + 3
                func_index = initial_func_count + 3
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.DoseLevel = 2500
                plan_optimization.Objective.ConstituentFunctions[func_index].DoseFunctionParameters.Weight = 8000

            logging.info("  ✓ Ring MaxDose 2500cGy (Weight: 8000)")
            objectives_added += 1
        except Exception as e:
            logging.error(f"  ✗ Ring MaxDose 추가 실패: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

        logging.info(f"총 {objectives_added}개 초기 Objective Functions 추가 완료")

        return objectives_added > 0

    except Exception as e:
        logging.error(f"초기 Objective Functions 추가 중 오류: {str(e)}")
        return False


def adjust_objectives_based_on_goals(plan, selected_goals, plan_optimization):
    """
    Clinical Goals 달성 여부에 따라 Objective Functions를 동적으로 조정
    개선: Weight 조정 + DoseLevel/PercentVolume 조정 + 미달성 Goal에 대한 새 Objective 자동 추가
    """
    try:
        objective = plan_optimization.Objective
        if objective is None or not hasattr(objective, 'ConstituentFunctions'):
            return False

        cf = objective.ConstituentFunctions
        adjustments_made = 0
        objectives_created = 0

        # Clinical Goals 평가 및 조정
        for goal in selected_goals:
            try:
                # Goal 정보 추출
                planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                goal_type = getattr(planning_goal, "Type", "Unknown")
                acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)
                parameter_value = getattr(planning_goal, "ParameterValue", None)

                current_value = None
                try:
                    current_value = goal.GetClinicalGoalValue()
                except:
                    continue

                # Goal 달성 여부 판단
                goal_met = False
                if current_value is not None and acceptance_level > 0:
                    try:
                        if goal_criteria == "AtMost":
                            goal_met = current_value <= acceptance_level
                        elif goal_criteria == "AtLeast":
                            goal_met = current_value >= acceptance_level
                    except:
                        pass

                roi_name = goal.ForRegionOfInterest.Name

                # 목표 미달성 시 처리
                if not goal_met:
                    # === STEP 1: 해당 ROI + 파라미터에 대한 Objective Function이 이미 있는지 확인 ===
                    # 같은 ROI라도 DoseLevel/PercentVolume이 다르면 별도 objective 필요
                    matching_objective_found = False

                    # DoseAtVolume goal인 경우 정확한 파라미터 매칭 확인
                    if goal_type == "DoseAtVolume":
                        target_dose = acceptance_level
                        target_volume = parameter_value * 100 if parameter_value < 1.0 else parameter_value

                        for func in cf:
                            try:
                                func_roi_name = func.ForRegionOfInterest.Name
                                if func_roi_name == roi_name:
                                    func_type = func.DoseFunctionParameters.FunctionType

                                    # MaxDvh 또는 MinDvh인 경우 파라미터 비교
                                    if func_type in ["MaxDvh", "MinDvh"]:
                                        func_dose = func.DoseFunctionParameters.DoseLevel
                                        func_volume = func.DoseFunctionParameters.PercentVolume

                                        # DoseLevel과 PercentVolume이 모두 유사하면 매칭
                                        dose_match = abs(func_dose - target_dose) < 10  # 10 cGy 허용 오차
                                        volume_match = abs(func_volume - target_volume) < 1.0  # 1% 허용 오차

                                        if dose_match and volume_match:
                                            matching_objective_found = True
                                            break
                            except:
                                continue
                    else:
                        # DoseAtVolume이 아닌 경우 기존 로직 (ROI 이름만 확인)
                        for func in cf:
                            try:
                                func_roi_name = func.ForRegionOfInterest.Name
                                if func_roi_name == roi_name:
                                    matching_objective_found = True
                                    break
                            except:
                                continue

                    # === STEP 2: 매칭되는 Objective가 없으면 동적으로 생성 ===
                    if not matching_objective_found:
                        if goal_type == "DoseAtVolume":
                            target_dose = acceptance_level
                            target_volume = parameter_value * 100 if parameter_value < 1.0 else parameter_value
                            logging.info(f"  🔧 {roi_name}에 {target_dose:.0f}cGy @ {target_volume:.1f}% Objective Function이 없습니다. 동적으로 생성합니다...")
                        else:
                            logging.info(f"  🔧 {roi_name}에 Objective Function이 없습니다. 동적으로 생성합니다...")

                        # Goal Type에 따라 적절한 Objective Function 생성
                        is_ptv = "CTV" in roi_name or "PTV" in roi_name

                        try:
                            initial_func_count = len(cf)

                            if goal_type == "DoseAtVolume":
                                # DoseAtVolume goal → MaxDVH for "AtMost", MinDVH for "AtLeast"
                                if goal_criteria == "AtMost":
                                    # OAR: V_dose <= X% → MaxDVH
                                    # For DoseAtVolume AtMost: acceptance_level = dose (cGy), parameter_value = volume (%)
                                    # Example: Ipsi_Lung V800cGy <= 15% → acceptance=800, parameter=0.15
                                    dose_level = acceptance_level  # This is the dose in cGy
                                    percent_volume = parameter_value * 100 if parameter_value < 1.0 else parameter_value  # Convert to %

                                    with CompositeAction(f'Add {roi_name} MaxDvh'):
                                        plan_optimization.AddOptimizationFunction(
                                            FunctionType="MaxDvh",
                                            RoiName=roi_name,
                                            IsConstraint=False,
                                            RestrictAllBeamsIndividually=False,
                                            RestrictToBeams=[],
                                            IsRobust=False,
                                            RestrictToBeamSet=None,
                                            UseRbeDose=False
                                        )
                                        func_index = initial_func_count
                                        cf[func_index].DoseFunctionParameters.DoseLevel = dose_level
                                        cf[func_index].DoseFunctionParameters.PercentVolume = percent_volume
                                        cf[func_index].DoseFunctionParameters.Weight = 5000

                                        logging.info(f"  ✅ {roi_name} MaxDvh 추가: {dose_level}cGy @ {percent_volume}% (Weight: 5000)")
                                        objectives_created += 1

                                elif goal_criteria == "AtLeast":
                                    # PTV: D_volume >= X cGy → MinDVH
                                    # For DoseAtVolume AtLeast: acceptance_level = dose (cGy), parameter_value = volume (%)
                                    # Example: CTV_WB D95% >= 2470cGy → acceptance=2470, parameter=0.95
                                    dose_level = acceptance_level  # This is the dose in cGy
                                    percent_volume = parameter_value * 100 if parameter_value < 1.0 else parameter_value  # Convert to %

                                    with CompositeAction(f'Add {roi_name} MinDvh'):
                                        plan_optimization.AddOptimizationFunction(
                                            FunctionType="MinDvh",
                                            RoiName=roi_name,
                                            IsConstraint=False,
                                            RestrictAllBeamsIndividually=False,
                                            RestrictToBeams=[],
                                            IsRobust=False,
                                            RestrictToBeamSet=None,
                                            UseRbeDose=False
                                        )
                                        func_index = initial_func_count
                                        cf[func_index].DoseFunctionParameters.DoseLevel = dose_level
                                        cf[func_index].DoseFunctionParameters.PercentVolume = percent_volume
                                        cf[func_index].DoseFunctionParameters.Weight = 30000

                                        logging.info(f"  ✅ {roi_name} MinDvh 추가: {dose_level}cGy @ {percent_volume}% (Weight: 30000)")
                                        objectives_created += 1

                            elif goal_type == "AverageDose":
                                # AverageDose "AtMost" → MaxEud
                                if goal_criteria == "AtMost":
                                    with CompositeAction(f'Add {roi_name} MaxEud'):
                                        plan_optimization.AddOptimizationFunction(
                                            FunctionType="MaxEud",
                                            RoiName=roi_name,
                                            IsConstraint=False,
                                            RestrictAllBeamsIndividually=False,
                                            RestrictToBeams=[],
                                            IsRobust=False,
                                            RestrictToBeamSet=None,
                                            UseRbeDose=False
                                        )
                                        func_index = initial_func_count
                                        cf[func_index].DoseFunctionParameters.DoseLevel = acceptance_level * 0.9
                                        cf[func_index].DoseFunctionParameters.Weight = 3000

                                        logging.info(f"  ✅ {roi_name} MaxEud 추가: {acceptance_level * 0.9:.0f}cGy (Weight: 3000)")
                                        objectives_created += 1

                            elif goal_type == "AbsoluteVolumeAtDose":
                                # Similar to DoseAtVolume but with absolute volume
                                if goal_criteria == "AtMost":
                                    dose_level = parameter_value if parameter_value else acceptance_level

                                    with CompositeAction(f'Add {roi_name} MaxDose'):
                                        plan_optimization.AddOptimizationFunction(
                                            FunctionType="MaxDose",
                                            RoiName=roi_name,
                                            IsConstraint=False,
                                            RestrictAllBeamsIndividually=False,
                                            RestrictToBeams=[],
                                            IsRobust=False,
                                            RestrictToBeamSet=None,
                                            UseRbeDose=False
                                        )
                                        func_index = initial_func_count
                                        cf[func_index].DoseFunctionParameters.DoseLevel = dose_level
                                        cf[func_index].DoseFunctionParameters.Weight = 3000

                                        logging.info(f"  ✅ {roi_name} MaxDose 추가: {dose_level}cGy (Weight: 3000)")
                                        objectives_created += 1

                        except Exception as e:
                            logging.error(f"  ✗ {roi_name} Objective 생성 실패: {str(e)}")
                            logging.error(traceback.format_exc())

                    # === STEP 3: 기존 Objective가 있으면 Weight 및 파라미터 조정 ===
                    else:
                        for i, func in enumerate(cf):
                            try:
                                func_roi_name = func.ForRegionOfInterest.Name

                                if func_roi_name != roi_name:
                                    continue

                                func_type = func.DoseFunctionParameters.FunctionType

                                # DoseAtVolume goal인 경우: 파라미터가 매칭되는 objective만 조정
                                if goal_type == "DoseAtVolume" and func_type in ["MaxDvh", "MinDvh"]:
                                    target_dose = acceptance_level
                                    target_volume = parameter_value * 100 if parameter_value < 1.0 else parameter_value

                                    func_dose = func.DoseFunctionParameters.DoseLevel
                                    func_volume = func.DoseFunctionParameters.PercentVolume

                                    # 파라미터가 매칭되지 않으면 skip
                                    dose_match = abs(func_dose - target_dose) < 10  # 10 cGy 허용 오차
                                    volume_match = abs(func_volume - target_volume) < 1.0  # 1% 허용 오차

                                    if not (dose_match and volume_match):
                                        continue  # 이 objective는 다른 goal을 위한 것

                                current_weight = func.DoseFunctionParameters.Weight

                                # PTV vs OAR 구분
                                is_ptv = "CTV" in roi_name or "PTV" in roi_name

                                # === 조정 전략 ===
                                with CompositeAction(f'Adjust {roi_name} Objective'):

                                    # 1. Weight 조정 (미달성 시 10% 증가)
                                    if is_ptv:
                                        weight_multiplier = 1.10  # 10% 증가 (기존 2.0에서 변경)
                                        max_weight = 200000
                                    else:
                                        weight_multiplier = 1.10  # 10% 증가 (기존 1.5에서 변경)
                                        max_weight = 100000

                                    new_weight = min(current_weight * weight_multiplier, max_weight)
                                    if new_weight != current_weight:
                                        func.DoseFunctionParameters.Weight = new_weight
                                        logging.info(f"  ↑ {roi_name} Weight: {current_weight:.0f} → {new_weight:.0f} (+10%)")
                                        adjustments_made += 1

                                    # 2. DoseLevel 및 PercentVolume 조정
                                    if func_type == "MinDvh" and is_ptv:
                                        # PTV MinDvh: D95% 목표가 낮으면 DoseLevel 5%씩 증가
                                        # 현재 goal과 func_type이 맞는지 확인
                                        if goal_type == "DoseAtVolume" and goal_criteria == "AtLeast" and current_value < acceptance_level:
                                            current_dose = func.DoseFunctionParameters.DoseLevel
                                            # DoseLevel을 5% 증가
                                            target_dose = current_dose * 1.05

                                            # CTV_WB의 경우 2645 cGy 이상으로 증가하지 않도록 제한
                                            if roi_name == "CTV_WB":
                                                target_dose = min(target_dose, 2645)
                                            else:
                                                # 다른 PTV는 acceptance_level × 1.05로 제한
                                                max_dose = acceptance_level * 1.05
                                                target_dose = min(target_dose, max_dose)

                                            if target_dose > current_dose:
                                                func.DoseFunctionParameters.DoseLevel = target_dose
                                                logging.info(f"  ↑ {roi_name} MinDvh DoseLevel: {current_dose:.0f} → {target_dose:.0f} cGy (+5%)")
                                                adjustments_made += 1

                                    elif func_type == "MaxDose":
                                        # PTV MaxDose: 현재값이 목표보다 높으면 DoseLevel 5% 감소 (PTV는 별도 goal)
                                        # OAR MaxDose: 현재값이 목표보다 높으면 DoseLevel 5% 감소
                                        if is_ptv:
                                            # PTV MaxDose의 경우 DoseAtAbsoluteVolume 또는 다른 MaxDose goal 확인
                                            if goal_criteria == "AtMost" and current_value > acceptance_level:
                                                current_dose = func.DoseFunctionParameters.DoseLevel
                                                # DoseLevel을 5% 감소
                                                target_dose = current_dose * 0.95
                                                # 너무 낮아지지 않도록 제한
                                                min_dose = acceptance_level * 0.90
                                                target_dose = max(target_dose, min_dose)

                                                # CTV_WB의 경우 2650 cGy 이하로 감소하지 않도록 제한
                                                if roi_name == "CTV_WB":
                                                    target_dose = max(target_dose, 2650)

                                                if current_dose > target_dose:
                                                    func.DoseFunctionParameters.DoseLevel = target_dose
                                                    logging.info(f"  ↓ {roi_name} MaxDose DoseLevel: {current_dose:.0f} → {target_dose:.0f} cGy (-5%)")
                                                    adjustments_made += 1
                                        elif not is_ptv and goal_criteria == "AtMost" and current_value > acceptance_level:
                                            current_dose = func.DoseFunctionParameters.DoseLevel
                                            # DoseLevel을 5% 감소
                                            target_dose = current_dose * 0.95
                                            # 목표보다 너무 낮아지지 않도록 제한
                                            min_dose = acceptance_level * 0.80
                                            target_dose = max(target_dose, min_dose)

                                            if current_dose > target_dose:
                                                func.DoseFunctionParameters.DoseLevel = target_dose
                                                logging.info(f"  ↓ {roi_name} MaxDose DoseLevel: {current_dose:.0f} → {target_dose:.0f} cGy (-5%)")
                                                adjustments_made += 1

                                    elif func_type == "MaxEud":
                                        # OAR MaxEud: 현재값이 목표보다 높으면 DoseLevel 5% 감소
                                        if not is_ptv and goal_criteria == "AtMost" and current_value > acceptance_level:
                                            current_dose = func.DoseFunctionParameters.DoseLevel
                                            # DoseLevel을 5% 감소
                                            target_dose = current_dose * 0.95
                                            # 목표보다 너무 낮아지지 않도록 제한
                                            min_dose = acceptance_level * 0.80
                                            target_dose = max(target_dose, min_dose)

                                            if current_dose > target_dose:
                                                func.DoseFunctionParameters.DoseLevel = target_dose
                                                logging.info(f"  ↓ {roi_name} MaxEud DoseLevel: {current_dose:.0f} → {target_dose:.0f} cGy (-5%)")
                                                adjustments_made += 1

                                    elif func_type == "MaxDvh":
                                        # OAR MaxDvh: DoseLevel과 PercentVolume을 5%씩 감소
                                        if not is_ptv and goal_type == "DoseAtVolume" and goal_criteria == "AtMost" and current_value > acceptance_level:
                                            # 1. DoseLevel을 5% 감소
                                            current_dose = func.DoseFunctionParameters.DoseLevel
                                            target_dose = current_dose * 0.95
                                            # 목표보다 너무 낮아지지 않도록 제한
                                            min_dose = acceptance_level * 0.80
                                            target_dose = max(target_dose, min_dose)

                                            if current_dose > target_dose:
                                                func.DoseFunctionParameters.DoseLevel = target_dose
                                                logging.info(f"  ↓ {roi_name} MaxDvh DoseLevel: {current_dose:.0f} → {target_dose:.0f} cGy (-5%)")
                                                adjustments_made += 1

                                            # 2. PercentVolume을 5% 감소 (예: 15% → 14.55%)
                                            current_volume = func.DoseFunctionParameters.PercentVolume
                                            target_volume = current_volume * 0.97
                                            # 너무 낮아지지 않도록 제한 (최소 1%)
                                            target_volume = max(target_volume, 1.0)

                                            if current_volume > target_volume:
                                                func.DoseFunctionParameters.PercentVolume = target_volume
                                                logging.info(f"  ↓ {roi_name} MaxDvh PercentVolume: {current_volume:.2f}% → {target_volume:.2f}% (-5%)")
                                                adjustments_made += 1

                            except Exception as e:
                                logging.debug(f"Function 조정 중 오류 (무시): {e}")
                                continue

            except Exception as e:
                logging.debug(f"Goal 평가 실패 (무시): {e}")
                continue

        if objectives_created > 0:
            logging.info(f"✅ 총 {objectives_created}개 새 Objective Functions 생성")
        if adjustments_made > 0:
            logging.info(f"✅ 총 {adjustments_made}개 파라미터 조정 완료")

        # ===================================================================
        # 강제 Weight 증가: Goals가 미달성이고 조정이 없을 경우 CTV_WB 강제 증가
        # ===================================================================
        if objectives_created == 0 and adjustments_made == 0:
            logging.info("조정할 파라미터 없음 - CTV_WB 강제 Weight 증가 시도")

            # CTV_WB goals 중 미달성 확인
            ctv_wb_unmet = False
            for goal in selected_goals:
                try:
                    roi_name = goal.ForRegionOfInterest.Name if hasattr(goal, 'ForRegionOfInterest') else ''
                    if roi_name == "CTV_WB":
                        planning_goal = goal.PlanningGoal if hasattr(goal, "PlanningGoal") else goal
                        goal_criteria = getattr(planning_goal, "GoalCriteria", "Unknown")
                        acceptance_level = getattr(planning_goal, "PrimaryAcceptanceLevel", 0)

                        current_value = None
                        try:
                            current_value = goal.GetClinicalGoalValue()
                        except:
                            pass

                        if current_value is not None and acceptance_level > 0:
                            if goal_criteria == "AtMost" and current_value > acceptance_level:
                                ctv_wb_unmet = True
                                break
                            elif goal_criteria == "AtLeast" and current_value < acceptance_level:
                                ctv_wb_unmet = True
                                break
                except:
                    continue

            # CTV_WB 미달성이면 강제 weight 증가
            if ctv_wb_unmet:
                logging.info("⚡ CTV_WB Goals 미달성 - 강제 Weight 증가")
                forced_adjustments = 0

                for func in cf:
                    try:
                        func_roi_name = func.ForRegionOfInterest.Name
                        if func_roi_name == "CTV_WB":
                            func_type = func.DoseFunctionParameters.FunctionType
                            current_weight = func.DoseFunctionParameters.Weight

                            # 강제 30% 증가 (더 공격적인 전략)
                            new_weight = min(current_weight * 1.30, 500000)  # 최대 500K

                            if new_weight > current_weight:
                                with CompositeAction(f'Force increase {func_roi_name} {func_type} weight'):
                                    func.DoseFunctionParameters.Weight = new_weight
                                logging.info(f"  ⚡ {func_roi_name} {func_type} Weight 강제 증가: {current_weight:.0f} → {new_weight:.0f} (+30%)")
                                forced_adjustments += 1
                    except Exception as e:
                        logging.debug(f"강제 weight 증가 실패: {e}")
                        continue

                if forced_adjustments > 0:
                    logging.info(f"✅ 총 {forced_adjustments}개 CTV_WB Objectives 강제 증가")
                    return True
                else:
                    logging.warning("⚠️ CTV_WB Objectives를 찾을 수 없음")
            else:
                logging.info("CTV_WB Goals 달성됨 - 강제 증가 불필요")

        return (objectives_created > 0) or (adjustments_made > 0)

    except Exception as e:
        logging.error(f"Objective 조정 중 오류: {e}")
        logging.error(traceback.format_exc())
        return False


def run_optimization_new_logic(plan, status_gui, selected_goals, start_time):
    """
    다이어그램 기반 최적화 로직:
    1. Start Block: Clinical Goals를 Objectives로 변환 (초기 weight: 2.9)
    2. Optimization: 50 iterations, Goal Gap 계산, Weights 동적 조정
       - Goals unmet: weight × 1.5-5
       - Goals met: weight × 0.8-0.9
    3. Optional Adjustment: 5 cycles 후 Dose level, Volume % 조정
    4. Goal Check: 모든 goals 충족 시 Scaling으로 이동
    5. Scaling: Target Dmax 27.81 Gy, Scale by 0.05 Gy
    """
    try:
        # PlanOptimizations와 BeamSets 존재 확인
        if len(plan.PlanOptimizations) == 0:
            logging.error("❌ PlanOptimizations가 없습니다!")
            messagebox.showerror("오류", "Plan에 Optimization이 설정되지 않았습니다.")
            return

        if len(plan.BeamSets) == 0:
            logging.error("❌ BeamSets이 없습니다!")
            messagebox.showerror("오류", "Plan에 BeamSet이 없습니다.")
            return

        plan_optimization = plan.PlanOptimizations[0]
        beam_set = plan.BeamSets[0]

        logging.info(f"Plan Optimization: {plan_optimization}")
        logging.info(f"BeamSet: {beam_set.DicomPlanLabel}")

        def check_goals_and_update_gui(cycle_num, operation_text):
            goals_achieved, met_goals_count = check_selected_goals(plan, selected_goals)
            current_max = get_ctv_wb_max_dose(plan)

            status_gui.update_status(
                cycle_num * 50,
                500,  # 최대 10 cycles
                met_goals_count,
                len(selected_goals),
                current_max,
                operation_text,
            )

            logging.info(f"{operation_text}: 달성된 목표 {met_goals_count}/{len(selected_goals)}, 전체 달성: {goals_achieved}")

            if goals_achieved:
                logging.info(f"{operation_text}에서 모든 목표 달성!")
                debug_goals_status(plan, selected_goals)

            return goals_achieved, current_max, met_goals_count

        logging.info("=== 다이어그램 기반 최적화 로직 시작 ===")
        logging.info("Start Block: Clinical Goals → Objectives (초기 weight: 2.9)")
        logging.info("Optimization: 50 iterations per cycle")
        logging.info("Goal Check 후 모두 충족 시 → Scaling")
        logging.info("충족하지 않으면 → Objective Functions 조정 및 재최적화")

        # Optimization 파라미터 설정: 50 iterations, intermediate dose, final dose
        try:
            logging.info("Setting optimization parameters...")

            # Max iterations 설정
            plan_optimization.OptimizationParameters.Algorithm.MaxNumberOfIterations = 50
            logging.info("✓ MaxNumberOfIterations = 50")

            # Convergence 설정 - 50회 iteration을 다 돌도록 설정
            try:
                # OptimalityTolerance를 매우 작게 설정하여 조기 종료 방지
                plan_optimization.OptimizationParameters.Algorithm.OptimalityTolerance = 1E-10
                logging.info("✓ OptimalityTolerance = 1E-10 (조기 종료 방지)")
            except:
                logging.warning("OptimalityTolerance 설정 실패 (무시)")

            # Dose calculation 설정
            plan_optimization.OptimizationParameters.DoseCalculation.ComputeIntermediateDose = True
            logging.info("✓ ComputeIntermediateDose = True")

            plan_optimization.OptimizationParameters.DoseCalculation.ComputeFinalDose = True
            logging.info("✓ ComputeFinalDose = True")

        except Exception as e:
            logging.error(f"파라미터 설정 중 오류: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

        # Optimization Cycle Loop with Adaptive Strategy (개선된 전략)
        min_cycles = 3  # 최소 cycle (기존 5에서 단축)
        max_cycles = 20  # 최대 cycle (기존 15에서 증가)
        early_scaling_threshold = 5  # 이 cycle 이후 조기 스케일링 가능 (7 → 5로 더 빠른 스케일링)
        goals_achieved = False

        # 수렴 감지 변수
        convergence_history = []  # (cycle, met_count, d95, max_dose)
        convergence_window = 3  # 최근 3 cycle 확인
        convergence_threshold = 0.5  # 0.5% 미만 개선 시 수렴으로 판단

        logging.info(f"Starting adaptive optimization cycles (min={min_cycles}, max={max_cycles})...")
        logging.info(f"Convergence detection: {convergence_window} cycles, threshold={convergence_threshold}%")
        logging.info(f"⚡ 개선된 전략: 초기 강한 weights, 빠른 조정 (10%), 유연한 cycles (3-20)")

        for cycle in range(1, max_cycles + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Optimization Cycle {cycle}/{max_cycles}")
            logging.info(f"{'='*60}")

            try:
                # Run Optimization (50 iterations)
                logging.info(f"Cycle {cycle}: About to run optimization (50 iterations)...")
                logging.info(f"Cycle {cycle}: Calling plan_optimization.RunOptimization()...")

                plan_optimization.RunOptimization()

                logging.info(f"Cycle {cycle}: ✓ RunOptimization() completed")

                # Compute Final Dose
                logging.info(f"Cycle {cycle}: Computing final dose...")
                beam_set.ComputeDose(ComputeBeamDoses=True, DoseAlgorithm="CCDose", ForceRecompute=True)

                logging.info(f"✓ Cycle {cycle} optimization complete")

            except Exception as e:
                error_msg = str(e)
                logging.error(f"✗ Cycle {cycle} optimization error: {error_msg}")

                if "objective function" in error_msg.lower() or "constituent" in error_msg.lower():
                    messagebox.showerror(
                        "최적화 실패",
                        f"Objective Functions 오류:\n{error_msg}\n\n"
                        "Plan → Optimization에서 확인하세요."
                    )
                else:
                    messagebox.showerror("최적화 오류", f"Cycle {cycle} 오류:\n{error_msg}")

                raise

            # Goal Check and Convergence Tracking
            goals_achieved, _, met_goals_count = check_goals_and_update_gui(
                cycle, f"Cycle {cycle} 완료 - Goal Check"
            )

            # 현재 cycle의 D95%와 MaxDose 기록 (Total Dose)
            try:
                current_d95 = get_ctv_wb_d95_total(beam_set)
                current_max = get_ctv_wb_max_dose(plan)
                convergence_history.append((cycle, met_goals_count, current_d95, current_max))
            except:
                convergence_history.append((cycle, met_goals_count, 0, 0))

            if goals_achieved:
                logging.info(f"✅ Cycle {cycle}에서 모든 Clinical Goals 달성!")
                break

            # Goals not met - Convergence Detection
            logging.info(f"⚠️ {met_goals_count}/{len(selected_goals)} Goals 달성")

            # 수렴 감지 (최소 cycle 이후부터)
            converged = False
            if cycle >= min_cycles and len(convergence_history) >= convergence_window:
                recent_history = convergence_history[-convergence_window:]

                # D95% 개선율 계산
                d95_improvements = []
                for i in range(1, len(recent_history)):
                    prev_d95 = recent_history[i-1][2]
                    curr_d95 = recent_history[i][2]
                    if prev_d95 > 0:
                        improvement = abs(curr_d95 - prev_d95) / prev_d95 * 100
                        d95_improvements.append(improvement)

                # Goal 달성 수 개선 확인
                met_count_improvements = []
                for i in range(1, len(recent_history)):
                    prev_met = recent_history[i-1][1]
                    curr_met = recent_history[i][1]
                    met_count_improvements.append(curr_met - prev_met)

                avg_d95_improvement = sum(d95_improvements) / len(d95_improvements) if d95_improvements else 0
                met_count_stagnant = all(imp == 0 for imp in met_count_improvements)

                if avg_d95_improvement < convergence_threshold and met_count_stagnant:
                    converged = True
                    logging.info(f"🔍 수렴 감지!")
                    logging.info(f"  최근 {convergence_window} cycles D95% 평균 개선: {avg_d95_improvement:.2f}%")
                    logging.info(f"  Goal 달성 수 정체: {met_count_stagnant}")

            # 조기 스케일링 시도 (수렴 감지 시 또는 threshold cycle 도달 시)
            if converged or (cycle >= early_scaling_threshold and met_goals_count >= len(selected_goals) - 2):
                if converged:
                    logging.info(f"⚡ Cycle {cycle}: 수렴 감지로 조기 스케일링 시도")
                else:
                    logging.info(f"⚡ Cycle {cycle}: {met_goals_count}/{len(selected_goals)} goals 달성 - 조기 스케일링 시도")

                # Smart Scaling 시도 (Total Dose)
                try:
                    current_d95 = get_ctv_wb_d95_total(beam_set)
                    current_max = get_ctv_wb_max_dose(plan)
                except:
                    current_d95 = 0
                    current_max = 0

                scaling_success, final_max, final_d95 = try_prescription_scaling_for_unmet_goals(
                    plan, beam_set, selected_goals, current_d95, current_max
                )

                if scaling_success:
                    logging.info(f"✅ Cycle {cycle}: Smart Scaling 성공 - 최적화 종료")
                    goals_achieved = True
                    break
                else:
                    if converged:
                        logging.info(f"⚠️ Smart Scaling 실패 - 추가 {max_cycles - cycle} cycles 진행")

                        # Heterogeneity가 너무 높으면 MaxDose DoseLevel 강제 감소
                        try:
                            heterogeneity = current_max / current_d95 if current_d95 > 0 else 0
                            if heterogeneity > 1.15:  # Heterogeneity 임계치 (1.18 → 1.15로 조기 개입)
                                logging.info(f"⚡ Heterogeneity 너무 높음 ({heterogeneity:.4f}) - CTV_WB DoseLevel 강제 조정")

                                objective = plan_optimization.Objective
                                if objective and hasattr(objective, 'ConstituentFunctions'):
                                    for func in objective.ConstituentFunctions:
                                        try:
                                            if func.ForRegionOfInterest.Name == "CTV_WB":
                                                func_type = func.DoseFunctionParameters.FunctionType

                                                # MaxDose DoseLevel 강제 감소
                                                if func_type == "MaxDose":
                                                    current_dose_level = func.DoseFunctionParameters.DoseLevel
                                                    # 8% 강제 감소 (더 공격적인 전략)
                                                    new_dose_level = current_dose_level * 0.92
                                                    # 2650 cGy 아래로는 안내려감
                                                    new_dose_level = max(new_dose_level, 2650)

                                                    if new_dose_level < current_dose_level:
                                                        with CompositeAction('Force decrease CTV_WB MaxDose DoseLevel'):
                                                            func.DoseFunctionParameters.DoseLevel = new_dose_level
                                                        logging.info(f"  ⚡ CTV_WB MaxDose DoseLevel 강제 감소: {current_dose_level:.0f} → {new_dose_level:.0f} cGy (-8%)")
                                                        patient_obj = get_current("Patient")
                                                        patient_obj.Save()

                                                # MinDVH DoseLevel 강제 증가 (D95%가 목표에서 멀 경우)
                                                elif func_type == "MinDVH":
                                                    # D95% 목표: 2470 cGy (Total dose)
                                                    d95_goal = 2470.0
                                                    d95_gap = d95_goal - current_d95

                                                    # D95%가 목표보다 50 cGy 이상 낮으면 MinDVH DoseLevel 증가
                                                    if d95_gap > 50:
                                                        current_dose_level = func.DoseFunctionParameters.DoseLevel
                                                        # 5% 증가 (MaxDose 제약 고려하여 보수적)
                                                        new_dose_level = current_dose_level * 1.05
                                                        # 2601 cGy 위로는 안올라감
                                                        new_dose_level = min(new_dose_level, 2601)

                                                        if new_dose_level > current_dose_level:
                                                            with CompositeAction('Force increase CTV_WB MinDVH DoseLevel'):
                                                                func.DoseFunctionParameters.DoseLevel = new_dose_level
                                                            logging.info(f"  ⚡ CTV_WB MinDVH DoseLevel 강제 증가: {current_dose_level:.0f} → {new_dose_level:.0f} cGy (+5%, D95% gap={d95_gap:.1f})")
                                                            patient_obj = get_current("Patient")
                                                            patient_obj.Save()
                                        except:
                                            continue
                        except Exception as e:
                            logging.debug(f"Heterogeneity 기반 조정 실패: {e}")
                    else:
                        logging.info(f"⚠️ Smart Scaling 불가능 - 추가 최적화 계속")

            logging.info(f"미달성 항목에 대해 Objective Functions 조정 중...")

            try:
                adjusted = adjust_objectives_based_on_goals(
                    plan, selected_goals, plan_optimization
                )
                if adjusted:
                    logging.info("✓ Objective Functions 조정 완료")
                    patient_obj = get_current("Patient")
                    patient_obj.Save()
                else:
                    logging.info("조정할 Objective Functions 없음")
            except Exception as e:
                logging.warning(f"Objective 조정 실패 (무시): {e}")

            # Optional Adjustment: 5 cycles 후 추가 조정
            if cycle % 5 == 0:
                logging.info(f"--- Optional Adjustment (Cycle {cycle}) ---")
                logging.info("Dose level 및 Volume % 재평가")
                # 필요시 추가 조정 로직 구현

        # Final Goal Check
        final_goals_achieved, final_met_count = check_selected_goals(plan, selected_goals)
        total_iterations = cycle * 50  # Calculate total iterations performed

        if not final_goals_achieved:
            logging.warning("=" * 60)
            logging.warning("⚠️ 모든 Clinical Goals가 달성되지 않았습니다!")
            logging.warning(f"달성: {final_met_count}/{len(selected_goals)} Goals")
            logging.warning("스케일링을 건너뜁니다.")
            logging.warning("=" * 60)

            # 최종 결과 메시지
            end_time = time.time()
            elapsed_sec = int(end_time - start_time) if start_time else 0
            hh = elapsed_sec // 3600
            mm = (elapsed_sec % 3600) // 60
            ss = elapsed_sec % 60
            final_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

            # 미달성 상태에서도 prescription scaling 시도
            logging.info("\n" + "="*60)
            logging.warning(f"⚠️ 일부 Goals 미달성 ({final_met_count}/{len(selected_goals)})")
            logging.info("Prescription Scaling을 시도하여 모든 goals 달성을 시도합니다.")
            logging.info("="*60)

            try:
                final_d95 = get_ctv_wb_d95_total(beam_set)
                final_max = get_ctv_wb_max_dose(plan)
            except:
                final_d95 = 0
                final_max = 0

            # Scaling으로 목표 달성 가능한지 시도 (Total Dose)
            scaling_success, final_max_scaled, final_d95_scaled = try_prescription_scaling_for_unmet_goals(
                plan, beam_set, selected_goals, final_d95, final_max
            )

            end_time = time.time()
            elapsed_sec = int(end_time - start_time) if start_time else 0
            hh = elapsed_sec // 3600
            mm = (elapsed_sec % 3600) // 60
            ss = elapsed_sec % 60
            final_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

            if scaling_success:
                completion_message = (
                    f"✅ Prescription Scaling으로 모든 Goals 달성!\n"
                    f"총 {cycle}개 cycles ({total_iterations}회 반복) + Prescription Scaling\n"
                    f"달성된 Goals: {len(selected_goals)}/{len(selected_goals)} (100%)\n"
                    f"CTV_WB D95: {int(final_d95_scaled)} cGy\n"
                    f"CTV_WB Max Dose: {int(final_max_scaled)} cGy\n"
                    f"총 소요시간: {final_time_str}"
                )
            else:
                completion_message = (
                    f"최적화 완료 (Scaling으로도 미달성)\n"
                    f"달성된 Goals: {final_met_count}/{len(selected_goals)}\n"
                    f"총 {cycle}개 cycles ({total_iterations}회 반복) 실행\n"
                    f"CTV_WB D95: {int(final_d95)} cGy\n"
                    f"CTV_WB Max Dose: {int(final_max)} cGy\n"
                    f"총 소요시간: {final_time_str}\n\n"
                    f"⚠️ 추가 최적화가 필요합니다."
                )

            if status_gui:
                status_gui.show_completion_message(
                    completion_message,
                    final_time=final_time_str,
                    final_d95=final_d95_scaled if scaling_success else final_d95,
                    final_max=final_max_scaled if scaling_success else final_max,
                )

            logging.info(f"전체 최적화 완료. 총 소요시간: {final_time_str}")
            logging.info(f"총 {cycle}개 cycles ({total_iterations}회 반복) 실행")
            if scaling_success:
                logging.info(f"달성된 Goals: {len(selected_goals)}/{len(selected_goals)} (Prescription Scaling 적용)")
            else:
                logging.info(f"달성된 Goals: {final_met_count}/{len(selected_goals)}")
            return

        # 모든 Goals 달성 → Scaling 진행
        logging.info("\n" + "="*60)
        logging.info("✅ 모든 Clinical Goals 달성!")
        logging.info(f"총 {cycle}개 cycles ({total_iterations}회 반복) 실행")
        logging.info("Scaling 단계로 진입합니다.")
        logging.info("="*60)

        _, current_max, _ = check_goals_and_update_gui(cycle, "Scaling 시작")

        scaling_success, final_max, final_d95 = run_scaling_optimization(
            plan, beam_set, selected_goals, status_gui
        )

        end_time = time.time()
        elapsed_sec = int(end_time - start_time) if start_time else 0
        hh = elapsed_sec // 3600
        mm = (elapsed_sec % 3600) // 60
        ss = elapsed_sec % 60
        final_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        if scaling_success:
            completion_message = (
                f"✅ 최적화 및 스케일링 완료!\n"
                f"총 {cycle}개 cycles ({total_iterations}회 반복) + 스케일링\n"
                f"달성된 Goals: {len(selected_goals)}/{len(selected_goals)} (100%)\n"
                f"CTV_WB D95: {int(final_d95)} cGy\n"
                f"CTV_WB Max Dose: {int(final_max)} cGy\n"
                f"D95/Target: {(final_d95/2470)*100:.1f}%\n"
                f"총 소요시간: {final_time_str}"
            )
        else:
            completion_message = (
                f"✅ 최적화 완료 (스케일링 한계)\n"
                f"총 {cycle}개 cycles ({total_iterations}회 반복) + 스케일링 시도\n"
                f"달성된 Goals: {len(selected_goals)}/{len(selected_goals)} (100%)\n"
                f"CTV_WB D95: {int(final_d95)} cGy\n"
                f"CTV_WB Max Dose: {int(final_max)} cGy\n"
                f"총 소요시간: {final_time_str}\n\n"
                f"⚠️ 추가 스케일링 조정 필요"
            )

        status_gui.show_completion_message(
            completion_message,
            final_time=final_time_str,
            final_d95=final_d95,
            final_max=final_max,
        )

        logging.info(f"전체 최적화 완료. 총 소요시간: {final_time_str}")
        logging.info(f"총 {cycle}개 cycles ({total_iterations}회 반복) 실행")
        logging.info(f"달성된 Goals: {len(selected_goals)}/{len(selected_goals)} (100%)")

    except Exception as e:
        logging.error(f"최적화 중 오류 발생: {str(e)}")
        if status_gui:
            status_gui.root.destroy()
        messagebox.showerror("Error", f"최적화 중 오류가 발생했습니다: {str(e)}")


def main():
    """메인 실행 함수"""
    global SCRIPT_START_TIME
    global selected_goals_global

    SCRIPT_START_TIME = time.time()

    logging.info("Script started")
    try:
        plan = get_current_plan()
        if not plan:
            return

        patient = get_current("Patient")
        patient_id = patient.PatientID

        logging.info(f"=== 환자 정보 ===")
        logging.info(f"Patient ID: {patient_id}") if LOG_PATIENT_IDENTIFIERS else logging.info("Patient identifiers suppressed")
        logging.info(f"Patient Name: {patient.Name}") if LOG_PATIENT_IDENTIFIERS else None
        logging.info(f"Plan Name: {plan.Name}")

        if not setup_prescription(plan):
            return

        # Step 1: Load Clinical Goals from template 'b_dm_g'
        logging.info("\n" + "="*60)
        logging.info("Step 1: Clinical Goals Template 로딩")
        logging.info("="*60)

        if not load_clinical_goals_from_template(plan, template_name=CLINICAL_GOAL_TEMPLATE):
            logging.error("Clinical Goals template 로딩 실패")
            messagebox.showerror("오류", "Clinical Goals template 'b_dm_g'를 찾을 수 없습니다.")
            return

        beam_set = plan.BeamSets[0]
        patient.Save()
        plan.SetCurrent()

        # Step 2: Initial Goal Check
        logging.info("\n" + "="*60)
        logging.info("Step 2: 초기 Goal Check")
        logging.info("="*60)

        selected_goals = get_clinical_goals_safely(plan)
        selected_goals_global = selected_goals

        if not selected_goals:
            messagebox.showwarning("경고", "Clinical Goals를 찾을 수 없습니다.")
            return

        logging.info(f"로드된 Clinical Goals: {len(selected_goals)}개")
        debug_all_goals_brief(plan)

        # Check if all goals are already met
        goals_achieved, met_goals_count = check_selected_goals(plan, selected_goals)
        logging.info(f"초기 Goal 달성 현황: {met_goals_count}/{len(selected_goals)}")

        if goals_achieved:
            logging.info("✅ 모든 Clinical Goals가 이미 충족되었습니다!")
            logging.info("작업을 종료합니다.")
            messagebox.showinfo("완료", "모든 Clinical Goals가 이미 충족되어 있습니다.\n추가 최적화가 필요하지 않습니다.")
            return

        logging.info("⚠️ 일부 Clinical Goals가 미충족 상태입니다.")
        logging.info("Objective Functions를 추가하고 최적화를 시작합니다.")

        # Step 3: Add Initial Objective Functions
        logging.info("\n" + "="*60)
        logging.info("Step 3: 초기 Objective Functions 추가")
        logging.info("="*60)

        if not add_initial_objectives(plan):
            logging.error("초기 Objective Functions 추가 실패")
            messagebox.showerror("오류", "Objective Functions 추가에 실패했습니다.")
            return

        patient.Save()

        logging.info(f"안전하게 {len(selected_goals)}개의 Clinical Goals 선택됨")

        # 색상 맵 설정
        try:
            case.CaseSettings.DoseColorMap.ColorTable = patient_db.LoadTemplateColorMap(
                templateName="Default_YCC",
                lockMode=None
            ).ColorMap.ColorTable
            case.CaseSettings.DoseColorMap.PresentationType = "Absolute"

            # Prescription이 있는 경우에만 ReferenceValue 설정
            if (hasattr(beam_set, 'Prescription') and beam_set.Prescription and
                hasattr(beam_set.Prescription, 'PrescriptionDoseReferences') and
                len(beam_set.Prescription.PrescriptionDoseReferences) > 0):
                total_prescription_dose = beam_set.Prescription.PrescriptionDoseReferences[0].DoseValue
                case.CaseSettings.DoseColorMap.ReferenceValue = total_prescription_dose
                logging.info(f"색상 맵 ReferenceValue 설정: {total_prescription_dose} cGy")
            else:
                # Prescription이 없으면 기본값 2470 cGy 사용
                case.CaseSettings.DoseColorMap.ReferenceValue = 2470
                logging.info("Prescription이 없어 기본값 2470 cGy 사용")
        except Exception as e:
            logging.warning(f"색상 맵 설정 중 오류 (계속 진행): {str(e)}")

        patient_info = {
            "patient_name": patient.Name if LOG_PATIENT_IDENTIFIERS and hasattr(patient, 'Name') else "",
            "patient_id": patient.PatientID if LOG_PATIENT_IDENTIFIERS and hasattr(patient, 'PatientID') else "",
            "plan_name": plan.Name if hasattr(plan, 'Name') else "",
        }

        status_gui = OptimizationStatusGUI(
            patient_name=patient_info["patient_name"],
            patient_id=patient_info["patient_id"],
            plan_name=patient_info["plan_name"],
            plan=plan,
            selected_goals=selected_goals
        )

        print("Optimization is starting...")
        run_optimization_new_logic(plan, status_gui, selected_goals, start_time=SCRIPT_START_TIME)

    except Exception as e:
        logging.error(f"Unexpected error in main function: {str(e)}")
        messagebox.showerror("Error", f"오류가 발생했습니다: {str(e)}")
    finally:
        end_time = time.time()
        elapsed_sec = int(end_time - SCRIPT_START_TIME)
        hh = elapsed_sec // 3600
        mm = (elapsed_sec % 3600) // 60
        ss = elapsed_sec % 60
        final_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        logging.info(f"Script execution completed. Total time: {final_time_str}")

        patient.Save()
        print("export_OPT_DOSE_N_PLAN_dicom()")
        export_dicom()

if __name__ == "__main__":
    main()
