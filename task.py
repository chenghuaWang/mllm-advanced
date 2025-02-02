import os
import re
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    format="%(asctime)s, [%(levelname)s]: %(message)s", level=logging.INFO
)

parser = argparse.ArgumentParser(description="Mllm task runner")
parser.add_argument("task_file", type=str, help="Path to task file")
args = parser.parse_args()
task_config = yaml.safe_load(Path(args.task_file).read_text())

PROJECT_ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def wildcard_to_regex(pattern):
    regex = re.escape(pattern)
    return f'^{regex.replace(r"\*", ".*").replace(r"\?", ".")}$'


def filter_files(directory, patterns, ignore_dirs=None, r=True, case_sensitive=True):
    root_dir = os.path.abspath(directory)

    ignore_abs = set()
    if ignore_dirs:
        for path in ignore_dirs:
            abs_path = os.path.normpath(os.path.join(root_dir, path))
            ignore_abs.add(abs_path.lower() if not case_sensitive else abs_path)

    flags = 0 if case_sensitive else re.IGNORECASE
    if isinstance(patterns, str):
        patterns = [patterns]
    regexes = [re.compile(wildcard_to_regex(p), flags) for p in patterns]

    matched = []

    if r:
        for root, dirs, files in os.walk(root_dir):
            current_path = os.path.normpath(root)
            compare_path = current_path.lower() if not case_sensitive else current_path

            if any(
                compare_path.startswith(ignored + os.sep) or compare_path == ignored
                for ignored in ignore_abs
            ):
                dirs[:] = []
                continue

            dirs[:] = [
                d
                for d in dirs
                if os.path.normpath(os.path.join(current_path, d)) not in ignore_abs
            ]

            for file in files:
                if any(rx.match(file) for rx in regexes):
                    matched.append(os.path.join(root, file))

    else:
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            if not os.path.isfile(full_path):
                continue

            file_path = os.path.normpath(full_path)
            compare_path = file_path.lower() if not case_sensitive else file_path
            if any(compare_path.startswith(ignored + os.sep) for ignored in ignore_abs):
                continue

            if any(rx.match(entry) for rx in regexes):
                matched.append(full_path)

    return matched


class Task:
    def __init__(self, config: Dict):
        self.config: Dict = config

    def make_command_str(self, commands: List) -> str:
        return " ".join(commands)

    def run(self):
        pass


class CMakeConfigTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.CMAKE_COMMAND = [
            "cmake",
            "-S",
            PROJECT_ROOT_PATH.as_posix(),
        ]

    def run(self):
        logging.info("CMake Config Task Start...")

        cmake_cfg_path = self.config.get("cmake_cfg_path", "build")
        self.CMAKE_COMMAND.extend(
            [
                "-G",
                "Ninja",
                "-B",
                os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path),
            ]
        )

        cmake_build_type = self.config.get("cmake_build_type", "Release")
        self.CMAKE_COMMAND.extend(["-DCMAKE_BUILD_TYPE=" + cmake_build_type])

        cmake_toolchain_file = self.config.get("cmake_toolchain_file", None)
        if cmake_toolchain_file:
            self.CMAKE_COMMAND.extend(
                [
                    "-DCMAKE_TOOLCHAIN_FILE=" + cmake_toolchain_file,
                ]
            )

        cmake_extra_args = self.config.get("cmake_extra_args", None)
        if cmake_extra_args:
            self.CMAKE_COMMAND.extend(cmake_extra_args)

        commands = self.make_command_str(self.CMAKE_COMMAND)
        logging.info(f"{commands}")
        os.system(commands)

        logging.warning(
            f'If you are using vscode to develop. Pls set `"clangd.arguments": ["--compile-commands-dir={os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path)}"]`'
        )

        logging.info("Finding targets in Ninja Builder:")
        os.system(
            self.make_command_str(
                [
                    "ninja",
                    "-C",
                    os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path),
                    "-t",
                    "targets",
                ]
            )
        )


class CMakeFormatTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        logging.info("CMake Format Task Start...")

        ignore_path = self.config.get("ignore_path", [])
        cmake_files = filter_files(
            PROJECT_ROOT_PATH, ["*.cmake", "CMakeLists.txt"], ignore_path
        )
        for file in cmake_files:
            logging.info(f"cmake-format {file} -o {file}")
            os.system(f"cmake-format {file} -o {file}")


class CMakeBuildTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.CMAKE_COMMAND = [
            "cmake",
            "--build",
            os.path.join(PROJECT_ROOT_PATH, self.config.get("cmake_cfg_path", "build")),
        ]

    def run(self):
        logging.info("CMake build Task Start...")
        targets = self.config.get("targets", None)
        if targets:
            for target in targets:
                sub_command = self.make_command_str(
                    self.CMAKE_COMMAND.extend(["--target", target])
                )
                logging.info(sub_command)
                os.system(sub_command)
        else:
            sub_command = self.make_command_str(self.CMAKE_COMMAND)
            logging.info(sub_command)
            os.system(sub_command)


TASKS = {
    "CMakeConfigTask": CMakeConfigTask,
    "CMakeFormatTask": CMakeFormatTask,
    "CMakeBuildTask": CMakeBuildTask,
}


if __name__ == "__main__":
    for task_name in task_config:
        TASKS[task_name](task_config[task_name]).run()
