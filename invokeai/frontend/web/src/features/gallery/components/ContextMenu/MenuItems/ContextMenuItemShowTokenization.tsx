import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { selectTokenizationDisplayMode } from 'features/gallery/store/gallerySelectors';
import { tokenizationDisplayModeChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold, PiCoinVerticalBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemShowTokenization = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const itemDTO = useItemDTOContext();
  const tokenizationDisplayMode = useAppSelector(selectTokenizationDisplayMode);

  const onClick = useCallback(() => {
    if (!tokenizationDisplayMode) {
      if (isImageDTO(itemDTO)) {
        dispatch(tokenizationDisplayModeChanged('positive'));
      } else {
        // TODO: Implement video select for compare
      }
    }
  }, [dispatch, itemDTO, tokenizationDisplayMode]);

  return (
    <MenuItem icon={<PiCoinVerticalBold />} onClickCapture={onClick}>
      {t('gallery.showTokenization')}
    </MenuItem>
  );
});

ContextMenuItemShowTokenization.displayName = 'ContextMenuItemShowTokenization';
