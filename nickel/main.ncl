let { build_example, ci, cleanup_pr_previews, deploy_reviewed_pr_preview, examples_on_main, pr_previews, release, reviewed_pr_preview, wasm_example_index_template, .. } = import kaaj in
let LIBRARY = "jonmo" in
let REPO = "https://github.com/databasedav/%{ LIBRARY }" in
let GITHUB_PAGES_URL = "https://databasedav.github.io/%{ LIBRARY }" in
let use_stable_toolchain = fun job =>
    job
    & {
        steps | force =
            std.array.map
                (fun step =>
                    if std.record.has_field "with" step && std.record.has_field "toolchain" step.with then
                        step & { with = { toolchain | force = "stable" } }
                    else
                        step
                )
                job.steps
    }
in
{
    build_example_ = build_example LIBRARY false "bevy/webgpu",
    ci_ =
        ci
        & {
            jobs = {
                clippy = use_stable_toolchain ci.jobs.clippy,
                docs = use_stable_toolchain ci.jobs.docs,
            }
        },
    cleanup_pr_previews_ = cleanup_pr_previews REPO,
    pr_previews_ = pr_previews,
    reviewed_pr_preview_ = reviewed_pr_preview,
    deploy_reviewed_pr_preview_ = deploy_reviewed_pr_preview REPO GITHUB_PAGES_URL,
    examples_on_main_ = examples_on_main REPO,
    release_ = release,
}
