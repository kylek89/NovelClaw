# Write in a Book

Open the [local Book library](http://127.0.0.1:8010/books) after starting Docker Compose. Create a Book, give it a title, and paste your brain dump. The Book owns its cast, chats, outlines, scenes, and earlier versions. The original run-based workspace is still available from the portal.

1. Open **Cast → Find cast in brain dump** to ask Claw for candidates. Edit names and profiles, select the people you want, and save them. **Develop with Claw** reopens a character’s conversation. Under a reply, **Select profile details** lets you decide which paragraphs to keep.
2. In **Storyboard → Plan chapters with Claw**, choose the chapter range and give it direction. Each outline goes on its own chapter page. Existing outlines require explicit replacement. Use **O** for the outline or **P** for prose.
3. On **Prose**, enter the scene’s POV, story day or time, and direction. Save the scene before asking Claw to append. To divide a passage, put the cursor between the two parts and choose **Split at cursor**. Review the result and move to the next scene when ready. Drafting pauses between scenes so you can revise a passage or change direction. Earlier saved text is available under **Previous prose versions** or **Previous outline versions**.
4. Under a character’s **Record a change**, select the source scene, quote the passage, and enter the change. Use **As of scene** to inspect the log. Generation uses manuscript order for these records. A quote removed from its source is flagged and withheld. Knowledge and emotional records in scene prose are scoped to the named POV. The NSFW lane includes designated private profiles and tagged changes. The brain dump and ordinary writing fields are shared story material.
5. **Back up Book** downloads a JSON copy containing writing, chats, profiles, continuity, and saved versions, including any private material. **Restore a copy** in the library creates a separate Book. Store these downloads privately. Model credentials are configured separately in **Manage models**.

Manual authoring and backup work immediately. AI requests use the provider and model configured for this installation. A Book request can write one scene or up to twenty consecutive outlines. If another edit lands while Claw is writing, its reply remains in chat for review. A provider interruption can require a billing check before starting another request.

The Docker setup binds the local preview services to loopback. Its normal host data directories persist when containers are recreated. The portable Book limit is 20 MB, including earlier versions and chats.

The interface and generation prompts use English. To permanently remove a Book, open **Book library**, choose **Delete Book** on its row, and confirm its title in the dialog. Download a backup first if you want to restore it later.

**Craft tools → Characters** and **Cast** open the same Book profiles. From the library, **Characters** asks which Book to open. Each tab keeps its Book in the URL. Model settings apply across Books. The separate run workspace has its own session memory, world and style tools.
