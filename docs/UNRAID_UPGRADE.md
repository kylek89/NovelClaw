# Upgrade stock NovelClaw on Unraid

Release `unraid-2026.11.27.3` provides one Linux/amd64 Docker image for the portal, MultiAgent and Claw Studio. The archive contains application code. Your existing Unraid appdata supplies the Books and settings.

## Get the release

In a new directory on the server, download and verify the image before touching the running installation. Allow space for the download, the image and a separate appdata backup.

```sh
set -e
release=https://github.com/kylek89/NovelClaw/releases/download/unraid-2026.11.27.3
curl -fLO "$release/novelclaw-linux-amd64.tar.gz"
curl -fLO "$release/SHA256SUMS"
sha256sum -c SHA256SUMS
docker load -i novelclaw-linux-amd64.tar.gz
docker image inspect novelclaw:unraid-2026.11.27.3 --format '{{.Os}}/{{.Architecture}} {{.Id}}'
```

To build from source instead, clone the pinned version into a new directory and build there. The release archive uses the standard web build. If the stock installation needs the optional local embedding libraries, build with its existing `INSTALL_VECTOR_DEPS=true` argument.

```sh
git clone --depth 1 --branch unraid-2026.11.27.3 https://github.com/kylek89/NovelClaw.git NovelClaw-unraid-2026.11.27.3
docker build -t novelclaw:unraid-2026.11.27.3 NovelClaw-unraid-2026.11.27.3
```

## Preserve the existing installation

1. Record the stock image IDs and retain them under rollback tags. Save the current Compose files or Unraid templates, service commands, working directories, ports, network, private environment, mount sources and destinations. Inspect actual container mounts. An empty bind directory can start healthy and still show an empty library.
2. Arrange a writing pause and let queued and running generation finish. Stop the stock services for a consistent snapshot. Back up the portal database and settings, both writing services' complete data and runs directories, both vector stores, all configured external storage and private environment files. Include SQLite sidecar files and anything stored only in a container's writable layer. Keep an untouched backup and make a separate working copy for rehearsal. Access to these backups should be private.
3. Rehearse with the new image on the working copy. Use distinct container names, ports and private network settings. Check the fully resolved mount paths, including the shared portal database, before starting. Give each running stack its own writable storage. Preserve the shared session secret, saved provider credentials and private access configuration.
4. Compare against the snapshot. Check all three services' `/healthz` endpoints (default ports 8010, 8011 and 8012), enter both apps through the portal, open a known chat, manuscript and memory record, and check older runs. Claw Studio gives a legacy conversation and its linked run the same Book. Standalone legacy runs get their own Books. Titles alone do not establish common ownership. Check two disposable Books for separate chats, cast, memory and prose. If data is missing or appears under the wrong Book, stop before switching production.
5. For production, use the existing stack identity and verified original mounts. If writing resumed after rehearsal, take a fresh stopped-service backup. Set the image for **all three services** to `novelclaw:unraid-2026.11.27.3`. The fork Compose file also accepts `NOVELCLAW_IMAGE=novelclaw:unraid-2026.11.27.3`. Stock Compose files or Unraid templates with fixed image names require editing each image reference. Preserve each service's individual command and working directory. For Compose, use the installation's existing project name and file arguments with `up -d --no-build --pull never --wait`. For templates, configure use of the loaded local image. Repeat the health and content checks before reopening writing.
6. For Lenovo inference, verify the Claw Studio container can reach your Lenovo on the configured host and ports. In **Models**, run **Test connection** for each local provider. Use the installation's own OpenRouter key if desired. The desktop's provider settings do not travel in this image.

## Rollback

Stop the upgraded services. Preserve the post-upgrade state separately if anyone wrote new material. Restore the matching pre-upgrade appdata, environment and stock images together, then restart the original stack and recheck the same content. Starting stock against a migrated database is an unverified rollback. Keep the backup until the author accepts the upgrade.
