-- mermaid.lua: Pandoc Lua filter to render mermaid code blocks to PNG via mmdc
-- Usage: pandoc --lua-filter=mermaid.lua ...

local system = require 'pandoc.system'

local function mermaid(code, filetype)
    return system.with_temporary_directory("mermaid", function(tmpdir)
        local infile = pandoc.path.join({tmpdir, "input.mmd"})
        local outfile = pandoc.path.join({tmpdir, "output." .. filetype})
        
        local f = io.open(infile, "w")
        f:write(code)
        f:close()
        
        os.execute(string.format(
            'mmdc -i "%s" -o "%s" -b transparent --scale 2 2>nul',
            infile, outfile
        ))
        
        local img = io.open(outfile, "rb")
        if img then
            local imgdata = img:read("*a")
            img:close()
            return imgdata
        end
        return nil
    end)
end

function CodeBlock(block)
    if block.classes[1] == "mermaid" then
        local imgdata = mermaid(block.text, "png")
        if imgdata then
            local fname = pandoc.sha1(block.text) .. ".png"
            pandoc.mediabag.insert(fname, "image/png", imgdata)
            return pandoc.Para{pandoc.Image({}, fname)}
        end
    end
end
