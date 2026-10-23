// Probe: archiver.  Program linked with the static library.
extern "C" int ulmblas_probe_lib();

int
main()
{
    return ulmblas_probe_lib() == 42 ? 0 : 1;
}
