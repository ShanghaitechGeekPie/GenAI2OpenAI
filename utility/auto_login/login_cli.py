import argparse
import json

from cas_login import LoginError, login_genai


def main():
    parser = argparse.ArgumentParser(description="ShanghaiTech GenAI CAS auto login utility")
    parser.add_argument("--credential", required=True, help="Credential in the format student_id@password")
    args = parser.parse_args()

    student_id, password = args.credential.split("@", 1)
    try:
        token = login_genai(student_id, password)
        print(json.dumps({"success": True, "token": token}, ensure_ascii=False))
    except LoginError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
