#ifndef ULMLAPACK_CHECKDEPS_H
#define ULMLAPACK_CHECKDEPS_H 1

//
//  Diagnostic build (-DULMLAPACK_CHECK_DEPS): every task records the tiles
//  it reads and writes and the start and end time of its body.  After the
//  parallel region check() verifies, for every tile, that any two accesses
//  of which at least one writes ran in the order in which the tasks were
//  created and did not overlap in time.  This tests that the OpenMP
//  runtime honors the depend clauses and that they are consistent with
//  the tiles the tasks actually access.  Without the macro everything here
//  compiles to nothing.
//

#ifdef ULMLAPACK_CHECK_DEPS

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ulmlapack {
namespace checkdeps {

struct Access
{
    long tile;
    bool write;
};

struct Record
{
    long id; // creation order
    const char *kind;
    long k, i, j;
    std::vector<Access> acc;
    double t0, t1;
};

struct Log
{
    std::vector<Record> rec;
    std::atomic<long> next;

    explicit Log(std::size_t reserve) : next(0)
    {
	rec.resize(reserve);
    }

    //  called by the generating thread when the task is created
    Record *
    create(const char *kind, long k, long i, long j)
    {
	long id = next.fetch_add(1);
	if (id >= (long)rec.size()) {
	    std::fprintf(stderr, "checkdeps: log too small\n");
	    std::abort();
	}
	Record &r = rec[id];
	r.id = id;
	r.kind = kind;
	r.k = k;
	r.i = i;
	r.j = j;
	r.acc.clear();
	r.t0 = r.t1 = -1;
	return &r;
    }
};

inline double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

//  Profile of the last call (diagnostic build): number of tasks, time to
//  create them, wall time of the parallel region, sum of the times of the
//  task bodies (the members of an LU panel team other than the panel
//  task itself are not included).
struct Profile
{
    long tasks = 0;
    int threads = 0;
    double generate = 0, wall = 0, busy = 0;
};

inline Profile &
lastProfile()
{
    static Profile p;
    return p;
}

inline void
profile(Log &log, int threads, double tStart, double tGenEnd, double tEnd)
{
    Profile &p = lastProfile();
    p.tasks = log.next.load();
    p.threads = threads;
    p.generate = tGenEnd - tStart;
    p.wall = tEnd - tStart;
    p.busy = 0;
    for (long r = 0; r < p.tasks; ++r) {
	p.busy += log.rec[r].t1 - log.rec[r].t0;
    }
}

//  returns the number of violations (and prints the first ones)
inline long
check(Log &log, const char *where)
{
    const long n = log.next.load();
    long bad = 0;
    // tile -> list of (id, write)
    std::vector<std::pair<long, std::pair<long, bool>>> all;
    for (long r = 0; r < n; ++r) {
	if (log.rec[r].t0 < 0 || log.rec[r].t1 < 0) {
	    if (++bad <= 5) {
		std::fprintf(stderr,
		             "checkdeps %s: task %ld (%s) did not run\n", where,
		             r, log.rec[r].kind);
	    }
	}
	for (const Access &a : log.rec[r].acc) {
	    all.push_back(std::make_pair(a.tile, std::make_pair(r, a.write)));
	}
    }
    std::sort(all.begin(), all.end());
    for (std::size_t s = 0; s < all.size();) {
	std::size_t e = s;
	while (e < all.size() && all[e].first == all[s].first) {
	    ++e;
	}
	// in creation order: every access must come after the last write
	// before it, and a write after all accesses before it
	long lastWrite = -1;
	double readsEnd = -1; // latest end of the reads since lastWrite
	long readsEndId = -1;
	for (std::size_t q = s; q < e; ++q) {
	    const long id = all[q].second.first;
	    const bool w = all[q].second.second;
	    const Record &r = log.rec[id];
	    if (lastWrite >= 0 && log.rec[lastWrite].t1 > r.t0) {
		if (++bad <= 5) {
		    const Record &p = log.rec[lastWrite];
		    std::fprintf(
		        stderr,
		        "checkdeps %s: tile %ld: %s(%ld,%ld,%ld)"
		        " started %.3g us before %s(%ld,%ld,%ld) ended\n",
		        where, all[q].first, r.kind, r.k, r.i, r.j,
		        (p.t1 - r.t0) * 1e6, p.kind, p.k, p.i, p.j);
		}
	    }
	    if (w) {
		if (readsEndId >= 0 && readsEnd > r.t0) {
		    if (++bad <= 5) {
			const Record &p = log.rec[readsEndId];
			std::fprintf(stderr,
			             "checkdeps %s: tile %ld: write "
			             "%s(%ld,%ld,%ld) overlaps read "
			             "%s(%ld,%ld,%ld)\n",
			             where, all[q].first, r.kind, r.k, r.i, r.j,
			             p.kind, p.k, p.i, p.j);
		    }
		}
		lastWrite = id;
		readsEnd = -1;
		readsEndId = -1;
	    } else if (r.t1 > readsEnd) {
		readsEnd = r.t1;
		readsEndId = id;
	    }
	}
	s = e;
    }
    return bad;
}

} // namespace checkdeps
} // namespace ulmlapack

#define CHECKDEPS_LOG(log, n) ulmlapack::checkdeps::Log log(n)
#define CHECKDEPS_CREATE(rec, log, kind, k, i, j)                              \
    ulmlapack::checkdeps::Record *rec = (log).create(kind, k, i, j)
#define CHECKDEPS_ACCESS(rec, tile, w) (rec)->acc.push_back({long(tile), w})
#define CHECKDEPS_START(rec) (rec)->t0 = ulmlapack::checkdeps::now()
#define CHECKDEPS_END(rec) (rec)->t1 = ulmlapack::checkdeps::now()
#define CHECKDEPS_CHECK(log, where) ulmlapack::checkdeps::check(log, where)
#define CHECKDEPS_TIME(t) double t = ulmlapack::checkdeps::now()
#define CHECKDEPS_TIME_SET(t) t = ulmlapack::checkdeps::now()
#define CHECKDEPS_PROFILE(log, th, t0, t1, t2)                                 \
    ulmlapack::checkdeps::profile(log, th, t0, t1, t2)

#else

#define CHECKDEPS_LOG(log, n)
#define CHECKDEPS_CREATE(rec, log, kind, k, i, j)
#define CHECKDEPS_ACCESS(rec, tile, w)
#define CHECKDEPS_START(rec)
#define CHECKDEPS_END(rec)
#define CHECKDEPS_CHECK(log, where) 0L
#define CHECKDEPS_TIME(t)
#define CHECKDEPS_TIME_SET(t)
#define CHECKDEPS_PROFILE(log, th, t0, t1, t2)

#endif

#endif // ULMLAPACK_CHECKDEPS_H
