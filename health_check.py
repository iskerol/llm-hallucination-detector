import os
import requests

def check():
    issues = 0
    print("--- PRE-DEPLOYMENT HEALTH CHECK ---\n")
    
    files_to_check = [
        "utils/__init__.py",
        "utils/nli.py",
        "utils/selfcheck.py",
        "utils/taxonomy.py",
        "models/faiss.index",
        "models/docs.pkl",
        "models/meta.pkl",
        ".github/workflows/ci.yml",
    ]
    
    for f in files_to_check:
        if os.path.exists(f):
            print(f"[{'READY':<7}] {f} exists")
        else:
            print(f"[{'MISSING':<7}] {f} does not exist")
            issues += 1

    # Check requirements.txt
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            reqs = f.read()
            missing_reqs = []
            for pkg in ["datasets", "spacy", "structlog"]:
                if pkg not in reqs:
                    missing_reqs.append(pkg)
            if missing_reqs:
                print(f"[{'MISSING':<7}] requirements.txt missing: {', '.join(missing_reqs)}")
                issues += 1
            else:
                print(f"[{'READY':<7}] requirements.txt has 'datasets', 'spacy', 'structlog'")
    else:
        print(f"[{'MISSING':<7}] requirements.txt does not exist")
        issues += 1

    # Check pipeline.py
    if os.path.exists("pipeline.py"):
        with open("pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "ag_news" in content:
                print(f"[{'MISSING':<7}] pipeline.py contains old 'ag_news' corpus")
                issues += 1
            else:
                print(f"[{'READY':<7}] pipeline.py does NOT contain 'ag_news'")
    else:
        print(f"[{'MISSING':<7}] pipeline.py does not exist")
        issues += 1

    # Check API reachable
    try:
        res = requests.get("http://localhost:8000/health", timeout=5.0)
        if res.status_code == 200:
            print(f"[{'READY':<7}] API reachable at localhost:8000/health")
        else:
            print(f"[{'MISSING':<7}] API returned status code {res.status_code}")
            issues += 1
    except Exception as e:
        print(f"[{'MISSING':<7}] API reachable at localhost:8000/health (Error: {e})")
        issues += 1
        
    print(f"\n--- SUMMARY ---")
    if issues == 0:
        print("Project is READY for deployment")
    else:
        print(f"Project has {issues} issues to fix before deployment")

if __name__ == "__main__":
    check()
