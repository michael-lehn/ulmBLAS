//
//  Driver of the numerical tests.
//
//      numeric_tests [--list] [FILTER...]
//
//  Runs all test cases (or those whose ID or name contains one of the
//  FILTER strings) and prints one line per test case:
//
//      PASS  Tnnnn name
//      FAIL  Tnnnn name: message
//      XFAIL Tnnnn name (#nnnn): message    known bug, expected to fail
//      XPASS Tnnnn name (#nnnn)             known bug not observed: an error
//      DESIGN Tnnnn name (#nnnn): observed  open design question, behavior
//                                           as recorded
//
//  followed by details of failures and a final line 'END'.  The exit status
//  is 0 if there was no FAIL and no XPASS.
//
//  --list prints the ID, the kind (TEST, XFAIL or DESIGN), the issue number
//  (or 0) and the name of each test case (used by test/run-tests.sh to
//  validate the registry).
//
#include <cstdio>
#include <cstring>
#include <exception>
#include <test/numeric/testing.h>

int
main(int argc, char **argv)
{
    using namespace test;

    std::vector<TestCase> &testCases = registry();

    if (argc>1 && std::strcmp(argv[1], "--list")==0) {
        for (std::size_t i=0; i<testCases.size(); ++i) {
            const TestCase &tc = testCases[i];
            std::printf("%s %s %d %s\n", tc.id,
                        tc.recorded ? "DESIGN" : tc.issue ? "XFAIL" : "TEST",
                        tc.issue, tc.name);
        }
        return 0;
    }

    int numErrors = 0;

    for (std::size_t i=0; i<testCases.size(); ++i) {
        const TestCase &tc = testCases[i];

        bool selected = argc<2;
        for (int a=1; a<argc && !selected; ++a) {
            selected = std::strstr(tc.id, argv[a])
                    || std::strstr(tc.name, argv[a]);
        }
        if (!selected) {
            continue;
        }

        Context ctx;
        try {
            tc.function(ctx);
        } catch (std::exception &e) {
            ctx.check(false, "exception: %s", e.what());
        } catch (...) {
            ctx.check(false, "unknown exception");
        }

        bool ok = ctx.numFailed==0 && ctx.numChecks>0;
        if (ctx.numChecks==0) {
            ctx.messages.push_back("no checks performed");
        }

        const char *status;
        if (tc.recorded) {
            if (ok && ctx.observed!=tc.recorded) {
                ctx.messages.push_back("observed '" + ctx.observed
                                       + "', recorded '" + tc.recorded + "'");
                ok = false;
            }
            status = ok ? "DESIGN" : "FAIL";
        } else if (tc.issue==0) {
            status = ok ? "PASS" : "FAIL";
        } else {
            status = ok ? "XPASS" : "XFAIL";
        }
        if (std::strcmp(status, "FAIL")==0 || std::strcmp(status, "XPASS")==0) {
            ++numErrors;
        }

        std::printf("%-5s %s %s", status, tc.id, tc.name);
        if (tc.issue) {
            std::printf(" (#%04d)", tc.issue);
        }
        if (tc.recorded && ok) {
            std::printf(": %s", ctx.observed.c_str());
        }
        if (!ok) {
            std::printf(": %s", ctx.messages.empty() ? ""
                                                     : ctx.messages[0].c_str());
            for (std::size_t m=1; m<ctx.messages.size(); ++m) {
                std::printf("\n      %s", ctx.messages[m].c_str());
            }
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    std::printf("END\n");
    return numErrors==0 ? 0 : 1;
}
