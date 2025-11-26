# 2025-ML-Final-Project# 2025-ML-Final

## Github 工作教學
以下是一份可以給組員看的 GitHub 開發教學範本，你可以直接貼到 README 或團隊文件裡用。

---

# GitHub 開發流程教學：每人一條分支的協作方式

為了避免大家直接在 `main`（或 `master`）上開發造成衝突、難以回溯，我們採用「每個人都在自己的分支開發，再合併回 main」的流程。

整體原則只有三條：

1. **不要直接在 `main` 上開發或 push。**
2. **每個人開發時都使用自己的分支（例如 `feature/xxx-yourname`）。**
3. **在合併回 `main` 之前，一定要先把最新的 `main` merge 回自己的分支，解決完衝突再發 PR。**

---

## 一、建立本地環境與 main 分支

第一次拿到專案時：

```bash
# 1. clone 專案
git clone <repo-url>
cd <repo-folder>

# 2. 確認目前在 main（或 master）
git branch
# 若不在 main，可切換：
git switch main

# 3. 把遠端最新 main 拉下來
git pull origin main
```

> 之後請把 `main` 當成「乾淨、可隨時部署」的分支，不在上面直接改動。

---

## 二、為自己建立專用開發分支

每個功能 / 作業 / issue，請 **從最新的 `main` 切出自己的分支**，命名建議：

* 功能：`feature/<功能名稱>-<你的名字>`
* 修 bug：`fix/<issue編號或簡短描述>-<你的名字>`

範例：

```bash
# 先確認在 main，且是最新的
git switch main
git pull origin main

# 再從 main 切出自己的分支
git switch -c feature/flow-matching-reproduce-deng
```

之後你所有的開發都在這個分支上完成。

---

## 三、開發時的日常操作：只 push 自己的分支

在自己的分支上開發時，基本循環是：

1. 修改程式
2. 查看變更
3. 加到 staging
4. commit
5. push 到「同名遠端分支」

範例：

```bash
# 確認目前所在分支
git branch   # 應該顯示 * feature/flow-matching-reproduce-deng

# 看有哪些變更
git status
git diff              # 看尚未 staged 的變更
git diff --cached     # 看已經 staged 的變更（如果有）

# 把需要的檔案加進這次 commit
git add file1.py file2.py
# 或一次加入目前所有變更（請小心使用）
git add .

# 寫 commit
git commit -m "Implement XXX and fix YYY"

# 第一次 push 這條分支到遠端
git push -u origin feature/flow-matching-reproduce-deng

# 之後在同一分支只要：
git push
```

重點：
**整個開發過程只 push 到自己的分支，不要 push 到 `main`。**

---

## 四、在合併回 main 前：先跟 main 對齊

當你覺得這個功能已經做完、測過，準備發 Pull Request (PR) 合併回 `main` 時，**請先把最新的 `main` 合併到你的分支**，確保衝突先在你這邊解決。

### 步驟：

1. 切回 `main`，拉遠端最新
2. 把 `main` merge 回自己的分支
3. 解決衝突、測試
4. 再 push 自己的分支

指令範例：

```bash
# 1. 切回 main，更新到最新
git switch main
git pull origin main

# 2. 回到自己的分支
git switch feature/flow-matching-reproduce-deng

# 3. 將 main 合併進來
git merge main

# 如果有衝突：
#   - 編輯有衝突的檔案
#   - 測試沒問題後
git add <有衝突的檔案...>
git commit   # 完成 merge commit

# 4. 把已經跟 main 對齊的分支 push 上去
git push
```

> 如此一來，等你發 PR 的時候，Review 者看到的是一條已經跟 `main` 對齊、沒有未知衝突的分支，合併會非常乾淨。

（如果你們團隊熟悉 rebase，也可以用 `git rebase main` 取代 `git merge main`，但教學給新手時通常先用 merge 比較直覺。詳細 rebase 觀念可參考 *Pro Git* 第三章「Rebasing」。）

---

## 五、發 Pull Request（PR）並合併

當你的分支已經：

* 功能完成
* 測試通過
* 已經把最新 `main` 合併進來、解決完衝突

就可以：

1. 到 GitHub 專案頁面
2. 點「Compare & pull request」或手動建立 PR
3. 選擇：

   * base branch：`main`
   * compare branch：你的分支（例如 `feature/flow-matching-reproduce-deng`）
4. 在 PR 描述中簡單說明：

   * 做了什麼
   * 有沒有 breaking change
   * 有沒有需要特別注意的地方
5. 等待 review、修正，最後由有權限的人把 PR 合併到 `main`。

合併完成後，如果這條分支已經不會再用，可以刪掉它：

* GitHub 頁面上有「Delete branch」按鈕
* 或在本機清一下：

```bash
# 刪除本地分支（已經合併到 main 後建議再刪）
git branch -d feature/flow-matching-reproduce-deng
```

---

## 六、流程重點總結（給大家背）

1. **永遠不要直接在 `main` 上開發或 push。**
2. **所有工作都從最新的 `main` 切出「自己的分支」。**
3. **開發期間只 push 到自己的分支。**
4. **要合併前，一定要先：`git pull origin main` → `git merge main` 到自己的分支，解決衝突再 push。**
5. **用 Pull Request 把自己的分支合併回 `main`。**



# run inference analysis
```
python analysis.py --output ./out_5   --checkpoint ./checkpoint-1799.pth     --total_samples 960     --batch_size 32       --step_size 0.2
```
output: output directory
step_size: inference step(0.1 -> 10 steps)