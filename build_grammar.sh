#!/usr/bin/env bash

JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
ANTLR_VERSION=$(antlr4 | awk '/Version /{print $NF}')

printf "\n\nJava Version $JAVA_VERSION\n"
printf "Antlr Version $ANTLR_VERSION\n"

printf "Generating Files from FhY Grammar...\n\n"
antlr4 -Dlanguage=Python3 -visitor grammar/FhY.g4
printf "Finished Building Files from FhY Grammar!\n"

printf "Now Moving Files into FhY Project\n"
mv grammar/*.py src/fhy_lang/lang/parser/
printf "Finished Moving FhY Parser Files!\n"

printf "Now Cleaning Up Files...\n"
rm grammar/*.interp grammar/*.tokens
printf "Finished Cleaning Up Files!\n"
