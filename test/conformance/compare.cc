//
//  compare ULMBLAS-OUTPUT REFERENCE-OUTPUT [EXPECTED]
//
//  Compares the outputs of a conformance test program linked with ulmBLAS
//  and with the reference (one line per call: "description: numbers",
//  complex numbers as re,im or (re,im)).  Numbers are equal if both are
//  NaN, both are the same infinity, or if they agree up to a relative
//  tolerance (2e-5 for s and c routines, 1e-12 for d and z; rounding of
//  solves); everything else, in particular guard values, must agree.
//  EXPECTED lists differences that are documented in test/known_issues.md,
//  one per line: "#NNNN regular-expression" for the descriptions of calls
//  that may differ.  Exit status 0 if all other calls agree.
//
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<double>
numbers(const std::string &s, bool &ok)
{
    std::vector<double> v;
    std::string tok;
    for (char c : s) {
        if (c==' ' || c=='|' || c==',' || c=='(' || c==')') {
            if (!tok.empty()) {
                char *end;
                v.push_back(std::strtod(tok.c_str(), &end));
                ok = ok && *end==0;
                tok.clear();
            }
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) {
        char *end;
        v.push_back(std::strtod(tok.c_str(), &end));
        ok = ok && *end==0;
    }
    return v;
}

bool
same(double a, double b, double tol)
{
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    if (std::isinf(a) || std::isinf(b)) {
        return a==b;
    }
    double m = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a-b)<=tol*m;
}

// type letter of a description such as "cblas_dgemv ..." or "ztrmm F ..."
double
tolerance(const std::string &desc)
{
    std::string name = desc.substr(0, desc.find(' '));
    if (name.compare(0, 6, "cblas_")==0) {
        name = name.substr(6);
    }
    if (name.compare(0, 1, "i")==0 && name.size()>1) {
        name = name.substr(1);
    }
    char t = name.empty() ? 'd' : name[0];
    return (t=='s' || t=='c') ? 2e-5 : 1e-12;
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc<3) {
        std::cerr << "usage: compare ULMBLAS-OUTPUT REFERENCE-OUTPUT "
                     "[EXPECTED]" << std::endl;
        return 2;
    }
    std::ifstream ulm(argv[1]), ref(argv[2]);
    if (!ulm || !ref) {
        std::cerr << "compare: cannot read " << argv[1] << " or " << argv[2]
                  << std::endl;
        return 2;
    }
    std::vector<std::pair<std::string, std::regex> > expected;
    if (argc>3) {
        std::ifstream ex(argv[3]);
        std::string line;
        while (std::getline(ex, line)) {
            if (line.empty() || line[0]!='#' || line.size()<7) {
                continue;
            }
            std::string issue = line.substr(0, 5);
            std::string re    = line.substr(line.find_first_not_of(' ', 5));
            expected.push_back(std::make_pair(issue, std::regex(re)));
        }
    }

    std::string                 u, r;
    long                        calls = 0, differ = 0;
    std::map<std::string, long> known, unexpected;
    std::vector<std::string>    examples;

    while (std::getline(ulm, u)) {
        if (!std::getline(ref, r)) {
            std::cerr << "compare: reference output is shorter" << std::endl;
            return 1;
        }
        std::string du = u.substr(0, u.find(':')), dr = r.substr(0, r.find(':'));
        if (du!=dr) {
            std::cerr << "compare: different calls:\n  " << du << "\n  " << dr
                      << std::endl;
            return 1;
        }
        ++calls;
        bool ok = true;
        std::vector<double> a = numbers(u.substr(du.size()+1), ok);
        std::vector<double> b = numbers(r.substr(dr.size()+1), ok);
        double tol = tolerance(du);
        bool equal = ok && a.size()==b.size();
        for (std::size_t i=0; equal && i<a.size(); ++i) {
            equal = same(a[i], b[i], tol);
        }
        if (equal) {
            continue;
        }
        std::string group = du.substr(0, du.find(' ', du.find(' ')+1));
        bool isKnown = false;
        for (const auto &e : expected) {
            if (std::regex_search(du, e.second)) {
                ++known[e.first + " " + group];
                isKnown = true;
                break;
            }
        }
        if (!isKnown) {
            ++differ;
            ++unexpected[group];
            if (examples.size()<5) {
                examples.push_back("ulmBLAS:   " + u.substr(0, 400) + "\n"
                                 + "reference: " + r.substr(0, 400));
            }
        }
    }
    if (std::getline(ref, r)) {
        std::cerr << "compare: ulmBLAS output is shorter" << std::endl;
        return 1;
    }
    std::cout << calls << " calls, " << differ << " differ";
    long k = 0;
    for (const auto &e : known) {
        k += e.second;
    }
    std::cout << " (and " << k << " known differences)" << std::endl;
    for (const auto &e : known) {
        std::cout << "  known " << e.first << ": " << e.second << std::endl;
    }
    for (const auto &e : unexpected) {
        std::cout << "  DIFFER " << e.first << ": " << e.second << std::endl;
    }
    for (const auto &e : examples) {
        std::cout << e << std::endl;
    }
    return differ==0 ? 0 : 1;
}
