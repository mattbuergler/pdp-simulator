#!/usr/bin/env python3

# author: SJP
# date: 2018-10
# credits: ratko

import os
import re
import time
import socket
import collections
import json
import numpy as np
import pathlib
import typing
import subprocess
import sys

main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'tools'))
try:
    from globals import *
    from TestDefinition import *
except ImportError:
    print('Failed to import nodules')
    raise

"""
    TestRunner Class Executing and Evalulating Tests
"""

class TestRunner:

    tests = []  # type: typing.List[pathlib.Path]
    timings = []  # type: typing.List[float]

    def __init__(self, config):
        self.velocity_tsa = config['velocity_tsa']
        self.bin = config['bin']
        self.tests = []  # type: typing.List[str]


    def findTests(self, directory: pathlib.Path):
        """
        searches current directory tree for test definitions
        """
        for directory in directory.rglob("testdef"):
            self.tests.append(directory.parent)
        self.tests.sort()

    def getNumTests(self):
        return len(self.tests)

    def displayNumTests(self):
        print((" found %d test(s)") % (len(self.tests)))

    def run(self):
        """
        run tests
        """
        errors = 0
        testcount = 0
        self.timings = []
        for testDirectory in self.tests:
            error = 0
            # check if test already passed
            if (testDirectory / "run" / "PASSED").exists():
                print(" test " + str(testDirectory) + " already passed last time")
                self.timings.append(0)
                continue

            self.setup(testDirectory)

            testDef = self.readTestDefinitions(testDirectory)

            testConf = json.loads((testDirectory / "testdef").read_bytes())
            # Run the SBG if necessary
            if testConf['SBG'] in ['all', 'timeseries', 'signal']:
                t, error = self.runSBG(testDirectory)
                self.timings.append(t)
            if testConf['RA'] == 'yes':
                t, error = self.runRA(testDirectory)
                self.timings.append(t)
            if testConf['eval'] == 'yes':
                t, error = self.runEval(testDirectory)
                self.timings.append(t)
            errors += self.compareData(testDirectory, testDef)
            testcount += len(testDef)

            if not errors:
                self.writePassed(testDirectory)

        self.showStats(errors, testcount)
        return errors

    def evaluate_only(self):
        """
        don't calculate tests, but just evaluate older runs
        """
        errors = 0
        testcount = 0
        for testDirectory in self.tests:
            # do the evaluation for both run and setup tests
            testDef = self.readDefinitions(testDirectory)
            errors += self.compareData(testDirectory, testDef)
            testcount += len(testDef)
        self.storeBenchmarks()
        self.showStats(errors, testcount)

    def update(self):
        """
        copy the values stored in a run/*.h5 container to ref/*.dat files
        all definitions can be found in the tesdef file
        """
        testDirectory = self.tests[0]
        print(
            " WARNING: testcases that refer to ANALYTICAL or EXPERIMENTAL values should NEVER be updated in this way..."
        )
        print(" be sure that this test has a NUMERICAL reference!")
        answer = input(" Updating %s... sure?\n [y/n]: " % testDirectory)
        if answer != "fuck yes":
            return
        # delete old files
        refDir = testDirectory / "ref"
        if len(list(refDir.iterdir())) > 0:
            delete_files_in_directory(refDir)
            print(" deleted old reference files")
        # update
        testDef = self.readDefinitions(testDirectory)
        for test in testDef:
            test.updateTest(testDirectory, refDir)

    def showStats(self, errors, testscount):
        if len(self.tests) > 0:
            if testscount > 0:
                ratio = float(errors) / float(testscount)
            else:
                ratio = 0
        else:
            ratio = -1
        if ratio < 0:
            msg = "there are no tests to run...probably you should ask dv :)"
        elif ratio == 0:
            msg = "a perfect day!"
        elif ratio < 0.05:
            msg = "almost o.k., i will manage that!"
        elif ratio < 0.1:
            msg = "well ... that will probably need you some time to fix it"
        elif ratio < 0.2:
            msg = "ouuuhhu ... that does not look good!"
        elif ratio < 0.5:
            msg = "hmm ... maybe you will have cancel your further plans ..."
        elif ratio < 0.8:
            msg = "did you ever thought about getting an easier job?"
        else:
            msg = "boahh ... you broke it completely!"

        print("\n ------------------------------------------------------ ")
        print(
            " succeeded in %d of %d tests: %s" % (testscount - errors, testscount, msg)
        )
        print(" ------------------------------------------------------ ")

    def cleanUp(self, what="hard"):
        """
        delete contents in directory run
        """
        for testDirectory in self.tests:
            runDirectory = testDirectory / "run"
            if not runDirectory.exists():
                continue
            if what == "hard":
                delete_files_in_directory(runDirectory)
            elif what == "soft":
                delete_file_if_exists(runDirectory / "PASSED")

    def readTestDefinitions(self, testDirectory):
        tests = []
        for test in self.readDefinitions(testDirectory):
            tests.append(test)
        return tests

    def readDefinitions(self, testDirectory: pathlib.Path):
        # read test definition file
        testDef = json.loads((testDirectory / "testdef").read_bytes())
        tags = {
            "dataset": None,
            "rel": -1,
            "abs": -1,
            "col": -1,
            "row": -1,
            "file": None,
        }
        # loop over all potential definitions
        tests = []
        for file in testDef['files']:
            myTags = tags.copy()
            for key in file.keys():
                myTags[key] = file[key]
            definition = TestDefinition(
                    myTags["tag"],
                    myTags["file"],
                    myTags["dataset"],
                    myTags["col"],
                    myTags["row"],
                    myTags["abs"],
                    myTags["rel"],
                    )
            tests.append(definition)
        if len(tests) == 0:
            PRINTERRORANDEXIT(
                "file testdef in "
                + str(testDirectory)
                + " does not contain any valid test!"
            )
        else:
            return tests
        return None

    def setup(self, testDirectory: pathlib.Path):
        """
        delete contents in run directory and copy files from input to run
        """
        create_dir_if_not_exists(testDirectory / "run")
        delete_files_in_directory(testDirectory / "run")
        copy_files(testDirectory / "input", testDirectory / "run")


    def runSBG(self, testDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        testDef = json.loads((testDirectory / "testdef").read_bytes())
        PRINTTITLE(" running SBG in %s " % str(testDirectory), "-")
        # get the filename of the runfile
        runfile = testDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "sbg.py")
        ]
        cmd.append("-r")
        cmd.append(testDef['SBG'])
        if self.velocity_tsa:
            cmd.append("-tsa")
        cmd.append(".")
        return self.runCommon(cmd, testDirectory, "a")

    def runRA(self, testDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running reconstruction in %s " % str(testDirectory), "-")
        # get the filename of the runfile
        runfile = testDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        # get a velocity estimate
        config = json.loads((runfile).read_bytes())
        vel = np.asarray(config["FLOW_PROPERTIES"]["mean_velocity"])
        mag_vel = np.sqrt(vel.dot(vel))
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "mssrc.py"),
            "-vel",
            str(mag_vel),
            "."
        ]
        return self.runCommon(cmd, testDirectory, "a")


    def runEval(self, testDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running evalutation in %s " % str(testDirectory), "-")
        # get the filename of the runfile
        runfile = testDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "evaluate.py"),
            testDirectory / "."
        ]
        return self.runCommon(cmd, testDirectory, "a")

    def runTSA(self, testDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running evalutation in %s " % str(testDirectory), "-")
        # get the filename of the runfile
        runfile = testDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "velcoity_tsa.py"),
            testDirectory / "."
        ]
        return self.runCommon(cmd, testDirectory, "a")

    def runCommon(
        self, cmd: typing.List[str], testDirectory: pathlib.Path, fileAccess="w"
    ):
        out = ""
        success = 0
        startingTime = time.time()
        try:
            out = run_process(cmd, testDirectory / "run").stdout
            success = 1
        except subprocess.CalledProcessError as error:
            out = error.stdout
        endTime = time.time()
        with open(str(testDirectory / "run" / "run.output"), fileAccess) as file:
            file.write(out)
        return endTime - startingTime, success

    def compareData(self, testDirectory: pathlib.Path, testDefinitions):
        """
        compare run and reference data
        """
        errors = 0
        for testDef in testDefinitions:
            errors += testDef.compare(testDirectory)
        return errors

    def writePassed(self, testDirectory: pathlib.Path):
        with open(str(testDirectory / "run" / "PASSED"), "w") as file:
            file.write("1")

    def getPassed(self):
        success = 0
        for testDirectory in self.tests:
            filename = testDirectory / "run" / "PASSED"
            if filename.exists():
                with open(str(filename), "r") as file:
                    if int(file.read()):
                        success += 1
        testcount = len(self.tests)
        with open(str(pathlib.Path("PASSED")), "w") as file:
            file.write(" %f" % (float(success) / float(testcount)))
        errors = testcount - success
        self.showStats(errors, testcount)
        return errors
